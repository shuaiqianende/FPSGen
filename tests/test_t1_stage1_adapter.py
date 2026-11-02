import hashlib,json
import torch
from torch import nn
import pytest
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow import t1_stage1_adapter as module

class FakeModel(nn.Module):
 def __init__(self,g,**kwargs):super().__init__();self.p=nn.Parameter(torch.ones(1));self.variant=kwargs['encoder_variant']

def test_t1_load_frozen_and_reject_wrong_identity(tmp_path,monkeypatch):
 g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
 cfg={'geometry':g.to_dict(),'model':{'encoder_variant':'patch_depth','objective':'fm','widths':[8,16,32,64],'input_feature_version':'coarse_f64_v1'}}
 c=tmp_path/'config.json';c.write_text(json.dumps(cfg));p=tmp_path/'model.pt';p.write_bytes(b'mock')
 digest=hashlib.sha256(json.dumps(json.loads(c.read_text()),sort_keys=True).encode()).hexdigest()
 value={'identity':{'config_sha256':digest,'geometry_sha256':g.sha256},'initialization_mode':'scratch','parameter_sync_pass':True,'successful_updates':50000,'sampler':{'successful_frame_exposures':100000},'ema':{'p':torch.tensor([3.])}}
 monkeypatch.setattr(module,'Stage1DirectOccupancy',FakeModel);monkeypatch.setattr(module.torch,'load',lambda *a,**k:value)
 adapter=module.FrozenT1Stage1Adapter(p,c,checkpoint_sha256=hashlib.sha256(b'mock').hexdigest())
 assert adapter.model.variant=='patch_depth' and adapter.model.p.item()==3 and not adapter.model.p.requires_grad and not adapter.training
 value['sampler']['successful_frame_exposures']=99998
 with pytest.raises(ValueError):module.FrozenT1Stage1Adapter(p,c,checkpoint_sha256=hashlib.sha256(b'mock').hexdigest())
