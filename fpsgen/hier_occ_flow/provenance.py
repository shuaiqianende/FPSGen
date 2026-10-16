"""Raw source content provenance is separate from historical Poisson pairing."""
import hashlib
import json
from pathlib import Path
from .storage import atomic_json,file_sha


def raw_source_snapshot(data_root,frames):
    data_root=Path(data_root); code=Path(__file__).resolve().parents[2]
    sources={}
    for seq in sorted({seq for seq,frame in frames}):
        directory=data_root/seq
        sources[seq]={name:{'sha256':file_sha(directory/name),'bytes':(directory/name).stat().st_size}
                      for name in ('map_clean.npy','poses.txt','calib.txt')}
    scans={f'{seq}_{frame:06d}':file_sha(data_root/seq/'velodyne'/f'{frame:06d}.bin') for seq,frame in frames}
    source={'definition':'Surface-hit occupancy before voxel/Poisson/fixed-count sampling',
        'map_pose_calibration':sources,'viewpoint_scan_sha256':scans,
        'source_code_sha256':{name:file_sha(code/'scripts'/name) for name in ('generate_semantickitti_poisson_gt.py','prepare_semantickitti.py')},
        'policy':{'world_sphere_radius':50.,'local_sphere_radius':50.,'open_z':[-4.,4.4],'viewpoint_voxel_size':10.}}
    sha=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    return source,sha


def bind_resolution_audit(output,data_root):
    output=Path(output); path=output/'resolution_audit.json'; audit=json.loads(path.read_text())
    frames=sorted({(r['sequence'],r['frame']) for r in audit['records']})
    snapshot,sha=raw_source_snapshot(data_root,frames)
    atomic_json(output/'raw_source_manifest.json',snapshot)
    audit.setdefault('paired_split_manifest_sha256',audit['source_manifest_sha256'])
    audit['source_manifest_sha256']=sha
    for dataset_manifest in output.glob('*/*/manifest.json'):
        value=json.loads(dataset_manifest.read_text()); value['source_manifest_sha256']=sha
        value['index_sha256']=file_sha(dataset_manifest.parent/value['index'])
        atomic_json(dataset_manifest,value)
    atomic_json(path,audit)
    return sha
