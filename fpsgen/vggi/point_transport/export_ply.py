"""Export the same full 180K point sets used in the quality report."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def rainbow_rgb(xyz):
    """Existing project palette: empirical height percentile, blue to red."""
    xyz = np.asarray(xyz)
    t = (rankdata(xyz[:, 2], method='average') - 1) / max(1, len(xyz)-1)
    stops = [0., 1/3, 2/3, 1.]
    colors = np.array([[0,0,255], [0,255,255], [255,255,0], [255,0,0]])
    return np.rint(np.stack([np.interp(t, stops, colors[:,i]) for i in range(3)],axis=1)).astype(np.uint8)


def write_ply(path, xyz, rgb=None, *, overwrite=False):
    xyz = np.asarray(xyz, dtype='<f4', order='C')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Finite Nx3 XYZ required')
    rgb = rainbow_rgb(xyz) if rgb is None else np.asarray(rgb)
    if rgb.shape != xyz.shape or rgb.dtype != np.uint8:
        raise ValueError('RGB must be uint8 Nx3')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    vertices = np.empty(len(xyz),dtype=dtype)
    for i,key in enumerate(('x','y','z')):vertices[key]=xyz[:,i]
    for i,key in enumerate(('red','green','blue')):vertices[key]=rgb[:,i]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment offline matching experiment; not model inference\n'
              'comment rainbow blue cyan yellow red; height percentile\n'
              f'element vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('xb') as f:
        f.write(header)
        f.write(vertices.tobytes())
    # Check the actual artifact, including preservation of source row order.
    with temporary.open('rb') as f:
        assert f.read(len(header)) == header
        restored = np.frombuffer(f.read(), dtype=dtype)
    np.testing.assert_array_equal(np.column_stack([restored[k] for k in ('x','y','z')]), xyz)
    np.testing.assert_array_equal(np.column_stack([restored[k] for k in ('red','green','blue')]), rgb)
    temporary.replace(path)


def export_assignment_scene(folder, source, target, permutation, tag, *, overwrite=False):
    """Use target height colors, transported consistently with each pairing."""
    np.testing.assert_array_equal(np.sort(permutation),np.arange(len(target)))
    folder.mkdir(exist_ok=True)
    gt_rgb=rainbow_rgb(target)
    tracked_rgb=gt_rgb[permutation]
    endpoint=target[permutation]
    clouds={'source_field_180k.ply':source,'gt_target_180k.ply':target,'matched_target_180k.ply':endpoint}
    for tau in (.25,.5,.75):clouds[f'tau{int(tau*100):03d}_180k.ply']=(1-tau)*source+tau*endpoint
    for name,xyz in clouds.items():
        if overwrite or not (folder/name).exists():
            write_ply(folder/name,xyz,gt_rgb if name=='gt_target_180k.ply' else tracked_rgb,overwrite=overwrite)
    (folder/'manifest.json').write_text(json.dumps(dict(scene=folder.parent.name,points=len(source),no_training=True,
        permutation=tag+'.permutation.npy',files=list(clouds),no_filter=True,
        color='GT height percentile blue/cyan/yellow/red; matched GT RGB follows source row through all tau values',
        interpretation='All XYZ preserved. Endpoint is a GT permutation; intermediate PLYs are linear matching trajectories, not model predictions.'),indent=2)+'\n')


def main():
    root = Path('fpsgen/vggi/runs/point_transport_sinkhorn_v1')
    for scene in ('00_000500', '00_001000', '01_000500'):
        folder = root/scene
        out = folder/'ply_quality_v1'
        out.mkdir(exist_ok=False)
        with np.load(folder/'clouds.npz') as f:
            source, target = f['source'], f['target']
        fast = np.load(folder/'online_v3_b64_i5000_e0.05_r12.permutation.npy')
        slow = np.load(folder/'round_none_b256_p16_permutation.npy')
        for ids in (fast, slow):
            np.testing.assert_array_equal(np.sort(ids), np.arange(len(target)))
        clouds = {'source_field_180k.ply':source, 'gt_target_180k.ply':target,
                  'fast_matched_target_180k.ply':target[fast],
                  'slow_matched_target_180k.ply':target[slow]}
        for tau in (.25, .5, .75):
            for name, ids in (('fast',fast), ('slow',slow)):
                clouds[f'{name}_tau{int(tau*100):03d}_180k.ply'] = (1-tau)*source+tau*target[ids]
        for name, xyz in clouds.items():
            write_ply(out/name, xyz)
        manifest = dict(scene=scene, count=len(source), no_training=True, no_threshold=True,
            input=str(folder/'clouds.npz'), fast_permutation='online_v3_b64_i5000_e0.05_r12.permutation.npy',
            slow_permutation='round_none_b256_p16_permutation.npy', files=list(clouds),
            interpretation='Matches quality_v1 inputs. Final fast/slow targets have identical GT geometry, different row correspondence. tau files are linear interpolations, not learned predictions.')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(out, flush=True)


if __name__ == '__main__': main()
