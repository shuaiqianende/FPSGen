import copy,torch
from torch import nn
from fpsgen.hier_occ_flow.stage2_fp16_migration import migrate_fp32_runtime

def test_optimizer_and_master_state_preserved():
 m=nn.Sequential(nn.Linear(3,4),nn.LayerNorm(4));opt=torch.optim.AdamW(m.parameters(),lr=0.0002)
 m(torch.randn(2,3)).square().mean().backward();opt.step();opt.zero_grad();ema=copy.deepcopy(m.state_dict());state=copy.deepcopy(opt.state_dict());before=copy.deepcopy(m.state_dict())
 masters,newopt,newema,_=migrate_fp32_runtime(m,opt,ema)
 assert all(p.dtype==torch.float16 for p in m.parameters())
 assert all(torch.equal(masters.params[n],before[n]) for n,_ in m.named_parameters())
 assert all(torch.equal(newema[n],v) for n,v in ema.items())
 for key,old in state['state'].items():
  for field,v in old.items():
   assert torch.equal(v,newopt.state_dict()['state'][key][field]) if torch.is_tensor(v) else v==newopt.state_dict()['state'][key][field]
