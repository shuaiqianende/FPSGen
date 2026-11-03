import copy,torch
from torch import nn
from fpsgen.hier_occ_flow.stage2_remove_score import remove_score
class Mock(nn.Module):
 def __init__(self):
  super().__init__();self.score=nn.Sequential(nn.Linear(1,16),nn.SiLU(),nn.Linear(16,16));self.fuse=nn.Sequential(nn.Linear(60,8),nn.SiLU(),nn.Linear(8,8))

def test_removal_preserves_neutral_function_and_adam():
 torch.manual_seed(11);m=Mock();opt=torch.optim.AdamW(m.parameters(),lr=.001)
 a=torch.randn(9,32);b=torch.randn(9,12);score=m.score(torch.ones(9,1));m.fuse(torch.cat((a,score,b),1)).square().mean().backward();opt.step()
 old=m.fuse(torch.cat((a,m.score(torch.ones(9,1)),b),1)).detach();ema=copy.deepcopy(m.state_dict());mom=opt.state[m.fuse[0].weight]['exp_avg'].clone()
 opt,ema=remove_score(m,opt,ema);new=m.fuse(torch.cat((a,b),1))
 assert not hasattr(m,'score') and m.fuse[0].in_features==44
 assert torch.allclose(old,new,atol=1e-6,rtol=1e-6)
 assert torch.equal(opt.state[m.fuse[0].weight]['exp_avg'],torch.cat((mom[:,:32],mom[:,48:]),1))
 assert not any(k.startswith('score.') for k in ema)
