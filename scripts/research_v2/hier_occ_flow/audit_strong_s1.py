"""Compare S1 to original on real LiDAR and audit its second-backbone gradients."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import argparse, json, torch, yaml
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_network import OccupancyConditionEncoder
from fpsgen.hier_occ_flow.strong_encoder_v2 import FullPyramidVoxDet
from fpsgen.hier_occ_flow.storage import SparseShardDataset, atomic_json
from fpsgen.hier_occ_flow.dataset import input_xyz

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());c['geometry']=yaml.safe_load((a.config.parent/c['geometry']).read_text());g=validate_config(c)
    from fpsgen.hier_occ_flow.stage1_network import OccupancyGridSpec
    spec=OccupancyGridSpec(x_min=g.origin[0],y_min=g.origin[1],z_min=g.origin[2],x_max=g.maximum[0],y_max=g.maximum[1],z_max=g.maximum[2],nx=384,ny=384,nz=32,xy_radius=g.sphere_radius,sphere_radius=g.sphere_radius)
    ds=SparseShardDataset(c['train_dataset'],g,verify=False);r=ds[0]
    xyz=torch.from_numpy(input_xyz(Path(c['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).cuda()
    torch.manual_seed(c['training']['seed']);ref=OccupancyConditionEncoder(spec,False).cuda().eval()
    s1=FullPyramidVoxDet(spec,False).cuda().eval();s1.load_state_dict(ref.state_dict(),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        baseline=ref(xyz);out=s1(xyz)
    if not torch.equal(baseline[0],out[0]): raise ValueError('S1 F1 does not match original')
    del baseline,out,ref
    s1.train();s1.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        out=s1(xyz); loss=out[1].float().square().mean()+out[2].float().square().mean()
    loss.backward()
    norms={}
    for name,module in [('local_backbone',s1.local_backbone),('triplane',s1.local_neck.sp_tpvs_cls)]:
        grads=[p.grad.float().norm() for p in module.parameters() if p.grad is not None]
        norms[name]=float(torch.linalg.vector_norm(torch.stack(grads))) if grads else 0.
        if not (0<norms[name]<float('inf')):raise ValueError('Missing finite F2/F3 gradient: '+name)
    atomic_json(a.output,dict(status='PASS',f1_exact=True,sequence=r['sequence'],frame=r['frame'],pyramid_shapes=[list(x.shape) for x in out],f2_f3_gradient_norms=norms))

if __name__=='__main__':main()
