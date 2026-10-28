"""Independent observed-LiDAR TriPlane encoders; no GT or VoxDet backbone."""
import torch
from torch import nn
from torch.nn import functional as F


def voxel_statistics(xyz, spec):
    """12D FP32 features and mask in B,C,Z,Y,X; geometry never uses autocast.

    Channels: log count, mean local XYZ, local XYZ variance, local Z min/max,
    normalized range mean/variance, observed flag. Empty cells are exactly zero.
    """
    from torch_scatter import segment_csr
    scenes=[xyz] if isinstance(xyz,torch.Tensor) else list(xyz)
    if not scenes:raise ValueError('At least one scene required')
    nx,ny,nz=(n//2 for n in spec.shape); rows=[]
    for points in scenes:
        if points.ndim!=2 or points.shape[1]!=3 or not bool(torch.isfinite(points).all()):raise ValueError('Finite XYZ required')
        with torch.autocast(points.device.type,enabled=False):
            p=points.float(); p=p[spec.valid_points(p)]
            if len(p)<2:raise ValueError('Too few observed points within ROI')
            origin=p.new_tensor(spec.origin);size=p.new_tensor(spec.size)*2
            q=torch.floor((p-origin)/size).long()
            if bool(((q<0)|(q>=q.new_tensor((nx,ny,nz)))).any()):raise ValueError('Voxel index outside grid')
            ids=q[:,0]+nx*(q[:,1]+ny*q[:,2]);n=nx*ny*nz
            center=origin+(q.float()+.5)*size;local=(p-center)/size
            radius=p.norm(dim=1,keepdim=True)/spec.sphere_radius
            order=torch.argsort(ids,stable=True);ids=ids[order];local=local[order];radius=radius[order]
            unique,counts=torch.unique_consecutive(ids,return_counts=True)
            ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
            count=counts.float().unsqueeze(1)
            sums=segment_csr(torch.cat((local,local.square(),radius,radius.square()),1),ptr,reduce='sum')
            mean=sums/count;variance=(mean[:,3:6]-mean[:,:3].square()).clamp_min(0)
            rvar=(mean[:,7:8]-mean[:,6:7].square()).clamp_min(0)
            zmin=segment_csr(local[:,2:3],ptr,reduce='min');zmax=segment_csr(local[:,2:3],ptr,reduce='max')
            feat=torch.cat((count.log1p(),mean[:,:3],variance,zmin,zmax,mean[:,6:7],rvar,torch.ones_like(count)),1)
            dense=p.new_zeros((n,12));dense[unique]=feat
            rows.append(dense.T.reshape(12,nz,ny,nx))
    dense=torch.stack(rows);return dense,dense[:,11:12]>0


def masked_projection(x,mask,dim):
    count=mask.float().sum(dim=dim).clamp_min(1)
    return (x*mask).sum(dim=dim)/count


class DepthwiseBlock(nn.Module):
    def __init__(self,inc,out,stride=1):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(inc,inc,3,stride,1,groups=inc,bias=False),nn.Conv2d(inc,out,1,bias=False),nn.GroupNorm(8,out),nn.SiLU())
    def forward(self,x):return self.net(x)


class PlanePyramid(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([DepthwiseBlock(12,32),DepthwiseBlock(32,48,2),DepthwiseBlock(48,64,2)])
        self.lateral=nn.ModuleList([nn.Conv2d(w,64,1) for w in (32,48,64)])
        self.refine=nn.ModuleList([DepthwiseBlock(64,64) for _ in range(3)])
    def features(self,x):
        a=[]
        for layer in self.layers:x=layer(x);a.append(x)
        return [layer(x) for layer,x in zip(self.lateral,a)]
    def finish(self,x):
        for i in (1,0):x[i]=x[i]+F.interpolate(x[i+1],size=x[i].shape[-2:],mode='bilinear',align_corners=False)
        return [f(v) for f,v in zip(self.refine,x)]


class TriPlaneEncoder(nn.Module):
    def __init__(self,spec,variant='v1'):
        super().__init__()
        if variant not in ('v1','v2','v3'):raise ValueError('Unknown TriPlane variant')
        if tuple(spec.shape)!=(384,384,32):raise ValueError('Expected original 384x384x32 parent grid')
        self.spec=spec;self.variant=variant;self.planes=PlanePyramid()
        self.readout=nn.ModuleList([nn.Conv2d(64,128,1) for _ in range(3)])
        if variant in ('v2','v3'):
            import spconv.pytorch as spconv
            self.sparse_in=nn.Linear(12,24);self.sparse_conv=spconv.SubMConv3d(24,24,3,padding=1,bias=False,indice_key='tri_observed')
            self.sparse_out=nn.Linear(24,128)
            self.bridge=nn.Sequential(nn.Conv3d(64,64,3,padding=1,groups=64,bias=False),nn.Conv3d(64,64,1,bias=False),nn.GroupNorm(8,64),nn.SiLU())
        if variant=='v3':
            self.gates=nn.ModuleList([nn.Conv3d(3,3,1) for _ in range(3)])
            for layer in self.gates:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.last_gate_stats=None

    def forward(self,xyz):
        x,mask=voxel_statistics(xyz,self.spec);b,_,z,y,w=x.shape
        xy=masked_projection(x,mask,2)
        grouped=self.variant in ('v2','v3')
        if grouped:
            xy=masked_projection(x.reshape(b,12,4,z//4,y,w),mask.reshape(b,1,4,z//4,y,w),3).permute(0,2,1,3,4).reshape(b*4,12,y,w)
        xz=masked_projection(x,mask,3);yz=masked_projection(x,mask,4)
        dtype=self.planes.layers[0].net[0].weight.dtype
        xy=xy.to(dtype);xz=xz.to(dtype);yz=yz.to(dtype)
        a=self.planes.features(xy);c=self.planes.features(xz);d=self.planes.features(yz)
        if grouped:
            coarse=a[2].reshape(b,4,64,*a[2].shape[-2:]).permute(0,2,1,3,4)
            bridge=self.bridge(coarse)
            a[2]=a[2]+bridge.permute(0,2,1,3,4).reshape_as(a[2])
            c[2]=c[2]+bridge.mean(3);d[2]=d[2]+bridge.mean(4)
        a=self.planes.finish(a);c=self.planes.finish(c);d=self.planes.finish(d)
        residual=None
        if grouped:
            import spconv.pytorch as spconv
            index=mask[:,0].nonzero().int();values=x.permute(0,2,3,4,1)[mask[:,0]].to(self.sparse_in.weight.dtype)
            # SpConv is not covered by PyTorch autocast: features and kernel
            # weights must explicitly match (FP32 AMP or explicit FP16 model).
            with torch.autocast(x.device.type,enabled=False):
                sparse=spconv.SparseConvTensor(self.sparse_in(values),index,[z,y,w],b)
                if sparse.features.dtype!=self.sparse_conv.weight.dtype:raise ValueError('Sparse feature/weight dtype mismatch')
                sparse=self.sparse_conv(sparse);sparse=sparse.replace_feature(self.sparse_out(F.silu(sparse.features)))
            residual=sparse.dense()
        outputs=[];stats=[]
        for i,(xyi,xzi,yzi) in enumerate(zip(a,c,d)):
            zi=z//(2**i);yi=y//(2**i);xi=w//(2**i)
            if grouped:
                xyvol=self.readout[i](xyi).reshape(b,4,128,yi,xi).permute(0,2,1,3,4)
                xyvol=F.interpolate(xyvol,size=(zi,yi,xi),mode='nearest')
            else:xyvol=self.readout[i](xyi).unsqueeze(2).expand(-1,-1,zi,-1,-1)
            xv=self.readout[i](xzi).unsqueeze(3).expand(-1,-1,-1,yi,-1)
            yv=self.readout[i](yzi).unsqueeze(4).expand(-1,-1,-1,-1,xi)
            if self.variant=='v3':
                density=F.adaptive_avg_pool3d(x[:,0:1],(zi,yi,xi));cover=F.adaptive_max_pool3d(mask.float(),(zi,yi,xi))
                yy=torch.linspace(-1,1,yi,device=x.device);xx=torch.linspace(-1,1,xi,device=x.device)
                dist=(yy[:,None].square()+xx[None,:].square()).sqrt().reshape(1,1,1,yi,xi).expand(b,-1,zi,-1,-1)
                gate=1+self.gates[i](torch.cat((dist,density,cover),1).to(self.gates[i].weight.dtype))
                # Preserve the exact V2 arithmetic at initialization; learn
                # only a zero-initialized residual modulation of that fusion.
                fused=(xyvol+xv+yv)/3+(xyvol*(gate[:,0:1]-1)+xv*(gate[:,1:2]-1)+yv*(gate[:,2:3]-1))/3
                if not self.training:
                    g=gate.detach().float();ranges={}
                    for name,low,high in (('near',0.,.4),('mid',.4,.8),('far',.8,2.)):
                        selected=((dist>=low)&(dist<high)).expand(b,1,zi,yi,xi)
                        ranges[name]=[float(g[:,j:j+1][selected].mean()) if bool(selected.any()) else None for j in range(3)]
                    stats.append(ranges)
            else:fused=(xyvol+xv+yv)/3
            if residual is not None:fused=fused+(residual if i==0 else F.avg_pool3d(residual,2**i)).to(fused.dtype)
            outputs.append(fused)
        self.last_gate_stats=stats if stats else None
        return tuple(outputs)
