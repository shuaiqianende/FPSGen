import torch
from torch import nn
from fpsgen.hier_occ_flow.stage2_no_lidar import remove_lidar_and_score
class Mock(nn.Module):
 def __init__(self):
  super().__init__();self.score=nn.Sequential(nn.Linear(1,16),nn.SiLU(),nn.Linear(16,16));self.lidar=nn.Sequential(nn.Linear(384,64),nn.SiLU(),nn.Linear(64,64));self.fuse=nn.Sequential(nn.Linear(144,8),nn.SiLU(),nn.Linear(8,8));self.condition_mode='pyramid_observed'
def test_removed_branches_and_shared_weights():
 torch.manual_seed(3);m=Mock();w=m.fuse[0].weight.detach().clone();opt=torch.optim.AdamW(m.parameters());opt=remove_lidar_and_score(m,opt)
 assert not hasattr(m,'score') and not hasattr(m,'lidar') and m.condition_mode=='observed'
 assert m.fuse[0].in_features==64
 assert torch.equal(m.fuse[0].weight,torch.cat((w[:,:32],w[:,112:]),1))
 assert {id(p) for group in opt.param_groups for p in group['params']}=={id(p) for p in m.parameters()}
