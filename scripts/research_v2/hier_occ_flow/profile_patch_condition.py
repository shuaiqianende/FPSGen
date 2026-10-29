"""Record model parameters, convolution/attention MACs and encoder CUDA latency."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch,yaml
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.patch_unet_condition import SpatialAttention
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.resources import gpu_preflight

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());c['geometry']=yaml.safe_load((a.config.parent/c['geometry']).read_text());g=validate_config(c);gpu=c['training']['physical_gpus'][0]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or gpu_preflight((gpu,),minimum_free_gib=16.,allow_shared_compute=False)['status']!='AVAILABLE':raise RuntimeError('GPU admission failed')
    torch.set_num_threads(4);torch.manual_seed(c['training']['seed']);ds=SparseShardDataset(c['train_dataset'],g,verify=False);r=ds[0]
    points=torch.from_numpy(input_xyz(Path(c['data_root'])/r['sequence']/'input_'/f"{r['frame']:06d}.npy")).cuda()
    model=Stage1DirectOccupancy(g,objective='fm',widths=c['model']['widths'],activation_checkpointing=False,encoder_variant=c['model']['encoder_variant']).cuda().eval()
    conv=[];att=[];hooks=[]
    def hook(name,module,args,out):
        n=out.numel(); kernel=module.weight[0].numel();conv.append(dict(name=name,shape=list(out.shape),macs=n*kernel))
    def attention(name,module,args,out):
        x=args[0];n=x.shape[2]*x.shape[3]*x.shape[4];att.append(dict(name=name,tokens=n,macs=2*x.shape[0]*n*n*x.shape[1]))
    for name,module in model.encoder.named_modules():
        if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):hooks.append(module.register_forward_hook(lambda m,i,o,n=name:hook(n,m,i,o)))
        if isinstance(module,SpatialAttention):hooks.append(module.register_forward_hook(lambda m,i,o,n=name:attention(n,m,i,o)))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):model.encoder(points)
    for h in hooks:h.remove()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for _ in range(3):model.encoder(points)
        torch.cuda.synchronize();times=[]
        for _ in range(5):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();model.encoder(points);end.record();end.synchronize();times.append(start.elapsed_time(end))
    atomic_json(a.output,dict(status='PASS',input=dict(sequence=r['sequence'],frame=r['frame']),parameters=sum(p.numel() for p in model.encoder.parameters()),convolution_macs=sum(x['macs'] for x in conv),attention_matmul_macs=sum(x['macs'] for x in att),scope='one frame batch1; excludes normalization, nonlinearities and index operations',conv=conv,attention=att,encoder_event_ms=times,peak_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
