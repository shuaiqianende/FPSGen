"""Sparse network components for FPSGen Approximate-OT Point Flow.

The PointFlow student parameterizes :math:`v_\psi` with a time-conditioned
Minkowski U-Net. At every encoder and decoder stage it fuses partial-LiDAR,
BEV-prior, semantic-layout, and flow-time features through learned gates.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from pykeops.torch import LazyTensor

__all__ = ['MinkGlobalEncIN', 'MinkUNetDiffIN']

class BasicConvBlock(nn.Module):
    """Convolution, batch normalization, and ReLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BEVEncoderFPN(nn.Module):
    """Lightweight feature pyramid encoder for BEV and layout conditioning maps."""
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 strides=(1, 2, 2, 2),
                 channels=(64, 64, 128, 256)):
        super(BEVEncoderFPN, self).__init__()

        assert len(strides) == len(channels), "strides and channels must have equal length"
        self.num_layers = len(strides)

        self.encoder_layers = nn.ModuleList()
        current_in_channels = in_channels

        for i in range(self.num_layers):
            stride = strides[i]
            out_ch = channels[i]

            stage_blocks = []
            stage_blocks.append(BasicConvBlock(current_in_channels, out_ch, kernel_size=3, stride=stride, padding=1))

            if i > 0:
                stage_blocks.append(BasicConvBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1))

            self.encoder_layers.append(nn.Sequential(*stage_blocks))
            current_in_channels = out_ch

        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for i in range(self.num_layers):
            self.lateral_convs.append(nn.Conv2d(channels[i], out_channels, kernel_size=1))

            if i < self.num_layers - 1:
                self.smooth_convs.append(BasicConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def _upsample_add(self, x, y):
        """Fuse adjacent FPN levels after bilinear upsampling."""
        _, _, H, W = y.size()
        upsampled_x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_x + y

    def forward(self, x):
        c_features = []
        for layer in self.encoder_layers:
            x = layer(x)
            c_features.append(x)

        p_features = []

        last_p = self.lateral_convs[-1](c_features[-1])
        p_features.append(last_p)

        for i in range(self.num_layers - 2, -1, -1):
            lat_c = self.lateral_convs[i](c_features[i])
            p_upsampled = self._upsample_add(p_features[-1], lat_c)
            p_smoothed = self.smooth_convs[i](p_upsampled)
            p_features.append(p_smoothed)

        p_features = p_features[::-1]

        return p_features

class MinkIN(nn.Module):
    """Apply instance normalization independently to each sparse scene."""

    def __init__(self, num_features):
        """Initialize normalization for ``num_features`` sparse channels."""
        nn.Module.__init__(self)
        self.num_features = num_features
        self.instance_norm = torch.nn.InstanceNorm1d(num_features, affine=True)

    def forward(self, x):
        coords = x.C.clone()
        feat = x.F.clone()

        bs = coords[:, 0].max() + 1

        if bs == coords.shape[0]:
            return x

        global_tensor = []
        for idx in range(bs):
            mask = (coords[:, 0] == idx)
            curr_tensor = feat[mask].reshape(1, -1, self.num_features).transpose(1, 2).contiguous()
            global_tensor.append(self.instance_norm(curr_tensor)[0].transpose(0, 1).contiguous())
        final_tensor = ME.SparseTensor(
            features=torch.cat(global_tensor, dim=0),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        return final_tensor

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BasicConvolutionBlockIN(nn.Module):
    """Sparse downsampling block using instance normalization for PointFlow."""
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    dimension=D),
            MinkIN(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class BasicDeconvolutionBlockIN(nn.Module):
    """Sparse decoder upsampling block using instance normalization."""
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc,
                                             outc,
                                             kernel_size=ks,
                                             stride=stride,
                                             dimension=D),
            MinkIN(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class ResidualBlockIN(nn.Module):
    """Instance-normalized sparse residual block used by the PointFlow U-Net."""
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    dimension=D),
            MinkIN(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=1,
                                    dimension=D),
            MinkIN(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                MinkIN(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class MinkGlobalEncIN(nn.Module):
    """Encode the active partial-LiDAR condition into sparse scene features."""
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [16, 16, 32, 64, 128, 64, 64, 48, 48]
        cs = [int(cr * x) for x in cs]
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            MinkIN(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            MinkIN(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlockIN(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlockIN(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlockIN(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlockIN(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Encode a partial sparse point cloud for PointFlow conditioning."""
        x0 = self.stem(x.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x4

class MinkUNetDiffIN(nn.Module):
    """Parameterize the Approximate-OT PointFlow velocity :math:`v_\psi`.

    The three-channel structural prior and two-channel semantic layout are
    encoded as feature pyramids. ``match_bev_to_sparse`` implements the
    paper's projection :math:`\Pi_{sp}`, while ``match_part_to_full`` transfers
    partial-LiDAR context to source voxels. Their features and the time
    embedding jointly gate all sparse U-Net stages.
    """
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [16, 16, 32, 64, 128, 64, 64, 48, 48]
        cs = [int(cr * x) for x in cs]
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)

        self.bev_encoder_main = BEVEncoderFPN(
            in_channels=3, out_channels=64, strides=(1, 2, 2, 2), channels=(16, 16, 32, 64)
        )
        self.bev_encoder_mask = BEVEncoderFPN(
            in_channels=2, out_channels=16, strides=(1, 2, 2, 2), channels=(16, 16, 32, 64)
        )

        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            MinkIN(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            MinkIN(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.latent_stage1 = nn.Sequential(
            nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2)
        )
        self.stage1_temp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2)
        )
        self.bev_main_latent_stage1 = nn.Sequential(
            nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.bev_mask_latent_stage1 = nn.Sequential(
            nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.latemp_stage1 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[0])
        )
        self.stage1 = nn.Sequential(
            BasicConvolutionBlockIN(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.latent_stage2 = nn.Sequential(
            nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2)
        )
        self.stage2_temp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2)
        )
        self.bev_main_latent_stage2 = nn.Sequential(
            nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.bev_mask_latent_stage2 = nn.Sequential(
            nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.latemp_stage2 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[1])
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlockIN(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        self.latent_stage3 = nn.Sequential(
            nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2)
        )
        self.stage3_temp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2)
        )
        self.bev_main_latent_stage3 = nn.Sequential(
            nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.bev_mask_latent_stage3 = nn.Sequential(
            nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.latemp_stage3 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[2])
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlockIN(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.latent_stage4 = nn.Sequential(
            nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2)
        )
        self.stage4_temp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2)
        )
        self.bev_main_latent_stage4 = nn.Sequential(
            nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.bev_mask_latent_stage4 = nn.Sequential(
            nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.latemp_stage4 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[3])
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlockIN(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlockIN(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlockIN(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.latent_up1 = nn.Sequential(nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2))
        self.up1_temp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2))
        self.bev_main_latent_up1 = nn.Sequential(nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))
        self.bev_mask_latent_up1 = nn.Sequential(nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))

        self.latent_up2 = nn.Sequential(nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2))
        self.up2_temp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2))
        self.bev_main_latent_up2 = nn.Sequential(nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))
        self.bev_mask_latent_up2 = nn.Sequential(nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))

        self.latent_up3 = nn.Sequential(nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2))
        self.up3_temp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2))
        self.bev_main_latent_up3 = nn.Sequential(nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))
        self.bev_mask_latent_up3 = nn.Sequential(nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))

        self.latent_up4 = nn.Sequential(nn.Linear(cs[4], cs[4]), nn.LeakyReLU(0.1, inplace=True), nn.Linear(cs[4], cs[4]//2))
        self.up4_temp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LeakyReLU(0.1, inplace=True), nn.Linear(self.embed_dim, cs[4]//2))
        self.bev_main_latent_up4 = nn.Sequential(nn.Linear(64, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))
        self.bev_mask_latent_up4 = nn.Sequential(nn.Linear(16, cs[4]//2), nn.LeakyReLU(0.1, inplace=True))

        self.latemp_up1 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[4]),
            nn.LeakyReLU(0.1, True),
            nn.Linear(cs[4], cs[4])
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlockIN(cs[4], cs[5], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlockIN(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1, D=self.D),
                ResidualBlockIN(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D)
            )
        ])

        self.latemp_up2 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[5]),
            nn.LeakyReLU(0.1, True),
            nn.Linear(cs[5], cs[5])
        )

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlockIN(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlockIN(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1, D=self.D),
                ResidualBlockIN(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D)
            )
        ])

        self.latemp_up3 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[6]),
            nn.LeakyReLU(0.1, True),
            nn.Linear(cs[6], cs[6])
        )

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlockIN(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlockIN(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1, D=self.D),
                ResidualBlockIN(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D)
            )
        ])

        self.latemp_up4 = nn.Sequential(
            nn.Linear(cs[4] * 2, cs[7]),
            nn.LeakyReLU(0.1, True),
            nn.Linear(cs[7], cs[7])
        )

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlockIN(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlockIN(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1, D=self.D),
                ResidualBlockIN(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D)
            )
        ])

        self.head = nn.Sequential(
            nn.Linear(cs[8], 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 3)
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_timestep_embedding(self, timesteps):
        """Create sinusoidal embeddings for continuous PointFlow times in [0, 1]."""
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be one-dimensional, got {timesteps.shape}")
        timesteps = timesteps * 1000.0
        half_dim = self.embed_dim // 2
        # Allocate frequencies on the caller's device. A hard-coded
        # ``torch.device('cuda')`` can target the wrong rank under DDP.
        exponent = torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        )
        emb = torch.exp(exponent * (-math.log(10000) / (half_dim - 1)))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def match_part_to_full(self, x_full, x_part):
        """Assign each current source voxel its nearest partial-LiDAR feature.

        Batch indices are separated in the KeOps distance space by scaling the
        batch coordinate beyond the spatial coordinate range, preventing
        matches across scenes.
        """
        full_c = x_full.C.clone().float()
        part_c = x_part.C.clone().float()
        max_coord = full_c.max()
        full_c[:, 0] *= max_coord * 2.
        part_c[:, 0] *= max_coord * 2.
        f_coord = LazyTensor(full_c[:, None, :])
        p_coord = LazyTensor(part_c[None, :, :])
        dist_fp = ((f_coord - p_coord) ** 2).sum(-1)
        match_feats = dist_fp.argKmin(1, dim=1)[:, 0]
        return x_part.F[match_feats]

    def match_bev_to_sparse(self, x_sparse, bev_feat):
        """Apply :math:`\Pi_{sp}` from dense BEV features to sparse voxels.

        Minkowski coordinates are converted back to metric XY using the
        0.05 m voxel size, projected into the ``[-50,50]`` BEV extent, and
        clamped at the boundary before indexed feature lookup.
        """
        voxel_size = 0.05
        pc_range = 50.0

        G = bev_feat.shape[2]
        coords = x_sparse.C.long()
        batch_idx = coords[:, 0]

        c_x = coords[:, 1].float()
        c_y = coords[:, 2].float()

        phys_x = c_x * voxel_size
        phys_y = c_y * voxel_size

        x_pixels = ((phys_x + pc_range) / (pc_range * 2.0) * G)
        y_pixels = ((phys_y + pc_range) / (pc_range * 2.0) * G)

        x_idx = torch.clamp(x_pixels.long(), 0, G - 1)
        y_idx = torch.clamp(y_pixels.long(), 0, G - 1)

        return bev_feat[batch_idx, :, x_idx, y_idx]

    def forward(self, img_main, img_mask, x, x_sparse, part_feats, t):
        """Predict :math:`v_\psi(\mathcal P_t,t,\hat B,C_m)`.

        ``img_main`` contains the complete ``[D,H,M]`` prior. At every U-Net
        scale, projected BEV/layout features, nearest partial-LiDAR features,
        and the continuous-time embedding form a multiplicative gate. The
        final sparse features are sliced back to the input TensorField order so
        the returned velocity remains source-indexed.
        """
        temp_emb = self.get_timestep_embedding(t)

        bev_main_feat = self.bev_encoder_main(img_main)[0]
        bev_mask_feat = self.bev_encoder_mask(img_mask)[0]

        x0 = self.stem(x_sparse)

        match0 = self.match_part_to_full(x0, part_feats)
        p0 = self.latent_stage1(match0)
        t0 = self.stage1_temp(temp_emb)
        batch_temp = torch.unique(x0.C[:, 0], return_counts=True)[1]
        t0 = torch.repeat_interleave(t0, batch_temp, dim=0)

        b_main0 = self.bev_main_latent_stage1(self.match_bev_to_sparse(x0, bev_main_feat))
        b_mask0 = self.bev_mask_latent_stage1(self.match_bev_to_sparse(x0, bev_mask_feat))
        w0 = self.latemp_stage1(torch.cat((p0, t0, b_main0, b_mask0), dim=-1))
        x1 = self.stage1(x0 * w0)

        match1 = self.match_part_to_full(x1, part_feats)
        p1 = self.latent_stage2(match1)
        t1 = self.stage2_temp(temp_emb)
        batch_temp = torch.unique(x1.C[:, 0], return_counts=True)[1]
        t1 = torch.repeat_interleave(t1, batch_temp, dim=0)

        b_main1 = self.bev_main_latent_stage2(self.match_bev_to_sparse(x1, bev_main_feat))
        b_mask1 = self.bev_mask_latent_stage2(self.match_bev_to_sparse(x1, bev_mask_feat))
        w1 = self.latemp_stage2(torch.cat((p1, t1, b_main1, b_mask1), dim=-1))
        x2 = self.stage2(x1 * w1)

        match2 = self.match_part_to_full(x2, part_feats)
        p2 = self.latent_stage3(match2)
        t2 = self.stage3_temp(temp_emb)
        batch_temp = torch.unique(x2.C[:, 0], return_counts=True)[1]
        t2 = torch.repeat_interleave(t2, batch_temp, dim=0)

        b_main2 = self.bev_main_latent_stage3(self.match_bev_to_sparse(x2, bev_main_feat))
        b_mask2 = self.bev_mask_latent_stage3(self.match_bev_to_sparse(x2, bev_mask_feat))
        w2 = self.latemp_stage3(torch.cat((p2, t2, b_main2, b_mask2), dim=-1))
        x3 = self.stage3(x2 * w2)

        match3 = self.match_part_to_full(x3, part_feats)
        p3 = self.latent_stage4(match3)
        t3 = self.stage4_temp(temp_emb)
        batch_temp = torch.unique(x3.C[:, 0], return_counts=True)[1]
        t3 = torch.repeat_interleave(t3, batch_temp, dim=0)

        b_main3 = self.bev_main_latent_stage4(self.match_bev_to_sparse(x3, bev_main_feat))
        b_mask3 = self.bev_mask_latent_stage4(self.match_bev_to_sparse(x3, bev_mask_feat))
        w3 = self.latemp_stage4(torch.cat((p3, t3, b_main3, b_mask3), dim=-1))
        x4 = self.stage4(x3 * w3)

        match4 = self.match_part_to_full(x4, part_feats)
        p4 = self.latent_up1(match4)

        t4 = self.up1_temp(temp_emb)
        batch_temp = torch.unique(x4.C[:, 0], return_counts=True)[1]
        t4 = torch.repeat_interleave(t4, batch_temp, dim=0)

        b_main4 = self.bev_main_latent_up1(self.match_bev_to_sparse(x4, bev_main_feat))
        b_mask4 = self.bev_mask_latent_up1(self.match_bev_to_sparse(x4, bev_mask_feat))

        cond4 = torch.cat((p4, t4, b_main4, b_mask4), dim=-1)

        w4 = self.latemp_up1(cond4)
        y1 = self.up1[0](x4 * w4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        match5 = self.match_part_to_full(y1, part_feats)
        p5 = self.latent_up2(match5)

        t5 = self.up2_temp(temp_emb)
        batch_temp = torch.unique(y1.C[:, 0], return_counts=True)[1]
        t5 = torch.repeat_interleave(t5, batch_temp, dim=0)

        b_main5 = self.bev_main_latent_up2(self.match_bev_to_sparse(y1, bev_main_feat))
        b_mask5 = self.bev_mask_latent_up2(self.match_bev_to_sparse(y1, bev_mask_feat))

        cond5 = torch.cat((p5, t5, b_main5, b_mask5), dim=-1)

        w5 = self.latemp_up2(cond5)
        y2 = self.up2[0](y1 * w5)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        match6 = self.match_part_to_full(y2, part_feats)
        p6 = self.latent_up3(match6)

        t6 = self.up3_temp(temp_emb)
        batch_temp = torch.unique(y2.C[:, 0], return_counts=True)[1]
        t6 = torch.repeat_interleave(t6, batch_temp, dim=0)

        b_main6 = self.bev_main_latent_up3(self.match_bev_to_sparse(y2, bev_main_feat))
        b_mask6 = self.bev_mask_latent_up3(self.match_bev_to_sparse(y2, bev_mask_feat))

        cond6 = torch.cat((p6, t6, b_main6, b_mask6), dim=-1)

        w6 = self.latemp_up3(cond6)
        y3 = self.up3[0](y2 * w6)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        match7 = self.match_part_to_full(y3, part_feats)
        p7 = self.latent_up4(match7)

        t7 = self.up4_temp(temp_emb)
        batch_temp = torch.unique(y3.C[:, 0], return_counts=True)[1]
        t7 = torch.repeat_interleave(t7, batch_temp, dim=0)

        b_main7 = self.bev_main_latent_up4(self.match_bev_to_sparse(y3, bev_main_feat))
        b_mask7 = self.bev_mask_latent_up4(self.match_bev_to_sparse(y3, bev_mask_feat))

        cond7 = torch.cat((p7, t7, b_main7, b_mask7), dim=-1)

        w7 = self.latemp_up4(cond7)
        y4 = self.up4[0](y3 * w7)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        feat = y4.slice(x).F

        pred_xyz = self.head(feat)

        return pred_xyz
