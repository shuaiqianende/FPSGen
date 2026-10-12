"""GPU parity of block checkpointing against the unchanged PWM reference adapter."""
from common import parser, read_config, configure_runtime, save_json
import torch
from fpsgen.voxel_point.point_flow import build_ptv3
from fpsgen.voxel_point.vendor.pwm.ptv3_reference_adapter import OfficialPTV3PointFlowNet


def main():
    args=parser(__doc__).parse_args()
    cfg,spec,sha=read_config(args.config)
    device,_=configure_runtime()
    torch.set_num_threads(4)
    options=cfg['network']
    torch.manual_seed(321)
    reference=OfficialPTV3PointFlowNet(in_channels=6,coordinate_scale=tuple((b-a)/2 for a,b in zip(spec.origin,spec.maximum)),flow_voxel_size=spec.size,**options).to(device).train()
    checkpointed=build_ptv3(spec,network=options).to(device).train()
    for name,p in reference.named_parameters():
        if 'time_modulation' in name:
            torch.nn.init.normal_(p,std=.01)
    checkpointed.load_state_dict(reference.state_dict())
    points=torch.rand(512,3,device=device)*torch.tensor([20.,20.,6.],device=device)-torch.tensor([10.,10.,3.],device=device)
    ids,_=spec.voxelize_points(points)
    centers=spec.voxel_centers(ids).to(device)
    t=torch.tensor([.4],device=device)
    results=[]
    gradients=[]
    for model in (reference,checkpointed):
        torch.manual_seed(987)
        with torch.autocast('cuda',dtype=torch.float16):
            predicted=model(points,centers,t)
            loss=predicted.float().square().mean()
        loss.backward()
        results.append(predicted.detach().float())
        gradients.append({name:p.grad.detach().float().clone() for name,p in model.named_parameters() if p.grad is not None})
    output_difference=float((results[0]-results[1]).abs().max())
    error=max(float((gradients[0][name]-gradients[1][name]).abs().max()) for name in gradients[0])
    finite=all(torch.isfinite(g).all() for group in gradients for g in group.values())
    report={'status':'PASS' if output_difference < .003 and error < .003 and finite else 'FAIL',
        'points':512,'purpose':'checkpoint numerical parity only; 180k gates are separate',
        'config_sha256':sha,'max_output_difference':output_difference,'max_gradient_difference':error,'finite':bool(finite)}
    save_json(args.output/'checkpoint_parity.json',report)
    print(report)
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
