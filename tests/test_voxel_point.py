"""Geometry, strict source allocation, sparse OT policies and inference contracts."""

import ast
import inspect
from pathlib import Path

import pytest
import torch

from fpsgen.voxel_point.voxel_spec import VoxelGridSpec
from fpsgen.voxel_point.count_allocator import allocate_counts, InvalidDensityError
from fpsgen.voxel_point.anchor_sampler import sample_gaussian_anchors
from fpsgen.voxel_point.density_loss import density_terms, count_sum_pool, occupied_run_lengths
from fpsgen.voxel_point.point_flow import flow_matching_pair, euler_rollout
from fpsgen.voxel_point.dataset import (development_split, load_xyz, paired_frames,
                                      VoxelPointDataset, DistributedSceneBatchSampler)
from fpsgen.voxel_point import pipeline


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def small_spec():
    return VoxelGridSpec(nx=16, ny=16, nz=8)


def test_xyz_round_trip_and_flat_dense_asymmetry():
    spec = VoxelGridSpec()
    index = torch.tensor([[17, 42, 3], [255, 127, 31], [0, 0, 0]])
    ids = spec.flatten_voxel_ids(index)
    assert ids[0] == (3*256+42)*256+17
    assert torch.equal(spec.unflatten_voxel_ids(ids), index)
    centers = spec.voxel_centers(ids)
    actual = ((centers.double()-torch.tensor(spec.origin))/torch.tensor(spec.size)).floor().long()
    assert torch.equal(actual, index)
    data = torch.zeros(spec.num_voxels)
    data[ids] = torch.tensor([2., 3., 4.])
    dense = spec.flat_to_dense(data)
    assert dense[17, 42, 3] == 2
    assert torch.equal(spec.dense_to_flat(dense), data)
    assert spec.size == pytest.approx((.390625, .390625, .2625))
    coarse = spec.voxel_position_grid_coarse()
    assert coarse.shape == (128,128,16,3)
    assert coarse[0,0,0].tolist() == pytest.approx([-49.609375,-49.609375,-3.7375])
    assert coarse[-1,-1,-1].tolist() == pytest.approx([49.609375,49.609375,4.1375])


def test_open_circle_and_z_boundary():
    spec = VoxelGridSpec()
    xyz = torch.tensor([[0,0,-4], [0,0,4.4], [50,0,0], [0,0,0], [float('nan'),0,0]])
    assert spec.valid_points(xyz).tolist() == [False,False,False,True,False]
    with pytest.raises(ValueError, match="Invalid GT"):
        spec.voxelize_points(xyz)
    mask = spec.valid_circle_mask()
    assert mask[128,128,0]
    assert not mask[0,0,0]
    # Find cell centers outside the disk whose cells intersect it.
    xy = torch.cartesian_prod(torch.arange(spec.nx),torch.arange(spec.ny))
    ids = spec.flatten_voxel_ids(torch.cat((xy,torch.zeros(len(xy),1,dtype=torch.long)),1))
    centers = spec.voxel_centers(ids)
    boundary = mask[:,:,0].flatten() & (centers[:,:2].square().sum(-1) >= 2500)
    assert boundary.any()
    # A real interior corner point in this outside-center cell is retained.
    cell = xy[boundary][0]
    lower = torch.tensor(spec.origin[:2])+cell*torch.tensor(spec.size[:2])
    upper = lower+torch.tensor(spec.size[:2])
    closest = torch.maximum(lower,torch.zeros(2))+torch.minimum(upper,torch.zeros(2))
    point = torch.cat((closest*.999999,torch.zeros(1)))[None]
    if spec.valid_points(point).all():
        voxel, _ = spec.voxelize_points(point)
        assert mask[tuple(spec.unflatten_voxel_ids(voxel)[0])]


def test_180k_gt_mass_and_strict_bad_count():
    spec = VoxelGridSpec()
    xyz = torch.zeros(180000,3)
    counts = spec.counts(xyz)
    assert counts.sum() == 180000
    assert counts.max() == 180000  # No invented count cap.
    with pytest.raises(ValueError,match="exactly"):
        spec.counts(xyz[:-1])


def test_gt_roi_filter_records_actual_mass_and_keeps_strict_default(tmp_path):
    import numpy as np
    points = np.zeros((180000, 3), dtype=np.float32)
    # Real failed frame 00/000447: the stored float32 point is just outside.
    points[0] = [49.28866958618164, 8.40399169921875, .021183490753173828]
    points[1] = [0., 0., 4.4]
    np.save(tmp_path/'gt.npy',points)
    np.save(tmp_path/'input.npy',points[2:5])
    frames = [{'sequence':'00','frame':447,'gt':str(tmp_path/'gt.npy'),
               'input':str(tmp_path/'input.npy')}]
    with pytest.raises(ValueError,match='Rejected scene'):
        VoxelPointDataset(frames,small_spec())[0]
    sample = VoxelPointDataset(frames,small_spec(),gt_roi_policy='filter')[0]
    assert sample['gt_xyz'].shape == (179998,3)
    assert sample['gt_counts'].sum() == 179998
    assert sample['gt_validation'] == {'roi_policy':'filter','original_points':180000,
                                      'retained_points':179998,'filtered_roi_points':2}
    counts = sample['gt_counts'][None,None].float()
    terms = density_terms(counts,counts,small_spec().valid_circle_mask()[None,None])
    assert all(value == 0 for value in terms.values())
    allocated = allocate_counts(sample['gt_counts'].float(),small_spec(),
                                generator=torch.Generator().manual_seed(7))
    assert allocated.sum() == 180000
    points[0,0] = np.nan
    np.save(tmp_path/'gt.npy',points)
    with pytest.raises(ValueError,match='Nonfinite'):
        VoxelPointDataset(frames,small_spec(),gt_roi_policy='filter')[0]
    np.save(tmp_path/'gt.npy',points[2:])
    with pytest.raises(ValueError,match='exactly 180000'):
        VoxelPointDataset(frames,small_spec(),gt_roi_policy='filter')[0]


def test_distributed_scene_batches_cover_epochs_without_rank_overlap():
    def stream(rank):
        return list(DistributedSceneBatchSampler(16,2,rank,2,8,shuffle=True,seed=42))
    first, second = stream(0), stream(1)
    assert first == stream(0) and second == stream(1)
    epochs = []
    for start in (0,4):
        epoch = []
        for batch in range(start,start+4):
            assert set(first[batch]).isdisjoint(second[batch])
            epoch.extend(first[batch]+second[batch])
        assert sorted(epoch) == list(range(16))
        epochs.append(epoch)
    assert epochs[0] != epochs[1]


def test_allocation_determinism_and_invalid_density():
    spec = small_spec()
    density = torch.rand(spec.shape)
    a = allocate_counts(density,spec,generator=torch.Generator().manual_seed(7))
    b = allocate_counts(density,spec,generator=torch.Generator().manual_seed(7))
    assert torch.equal(a,b)
    assert a.sum() == 180000 and a.min() >= 0
    assert (a[~spec.dense_to_flat(spec.valid_circle_mask())] == 0).all()
    for bad in (torch.zeros(spec.shape),torch.full(spec.shape,float('nan')),torch.full(spec.shape,float('inf'))):
        with pytest.raises(InvalidDensityError):
            allocate_counts(bad,spec,generator=torch.Generator().manual_seed(0))


def test_180k_truncated_gaussian_cell_circle_and_source_order():
    spec = VoxelGridSpec()
    # Deliberately place almost all mass in boundary-intersecting cells.
    xy = torch.cartesian_prod(torch.arange(256),torch.arange(256))
    ids = spec.flatten_voxel_ids(torch.cat((xy,torch.full((len(xy),1),16)),1))
    center = spec.voxel_centers(ids)
    boundary = spec.valid_circle_mask()[:,:,16].flatten() & (center[:,:2].square().sum(-1)>=2500)
    density = torch.zeros(spec.shape)
    density[xy[boundary][:,0],xy[boundary][:,1],16] = 1
    counts = allocate_counts(density,spec,generator=torch.Generator().manual_seed(7))
    a, aid = sample_gaussian_anchors(spec,counts,generator=torch.Generator().manual_seed(8))
    b, bid = sample_gaussian_anchors(spec,counts,generator=torch.Generator().manual_seed(8))
    assert a.shape == (180000,3) and aid.shape == (180000,)
    assert torch.equal(a,b) and torch.equal(aid,bid)
    actual,valid = spec.voxelize_points(a)
    assert valid.all() and torch.equal(actual,aid)
    assert (aid[1:] >= aid[:-1]).all()
    assert torch.unique(a,dim=0).shape[0] == 180000


def test_fp32_loss_mask_and_count_sum_pool():
    spec = small_spec()
    counts = torch.zeros(1,1,*spec.shape)
    counts[0,0,8,8,4] = 10
    predicted = torch.ones_like(counts,requires_grad=True)
    valid = spec.valid_circle_mask()[None,None]
    terms = density_terms(predicted,counts,valid)
    assert count_sum_pool(counts).sum() == 10
    sum(terms.values()).backward()
    assert torch.isfinite(predicted.grad).all()
    assert (predicted.grad[~valid] == 0).all()
    assert all(v.dtype == torch.float32 for v in terms.values())


def test_cubic_point_voxel_iou_matches_histogram_and_handles_negative_coordinates():
    import numpy as np
    from fpsgen.voxel_point.metrics import point_voxel_occupancy_metrics
    rng = np.random.default_rng(42)
    p,g = rng.uniform(-2,2,(200,3)),rng.uniform(-2,2,(200,3))
    metric = point_voxel_occupancy_metrics(torch.from_numpy(p),torch.from_numpy(g),voxel_size=.5)
    hp = np.histogramdd(p,bins=(8,8,8),range=((-2,2),)*3)[0] > 0
    hg = np.histogramdd(g,bins=(8,8,8),range=((-2,2),)*3)[0] > 0
    assert metric['iou'] == pytest.approx((hp & hg).sum()/(hp | hg).sum())
    negative = point_voxel_occupancy_metrics(torch.tensor([[-.1,0.,0.]]),torch.tensor([[.1,0.,0.]]))
    assert negative['iou'] == 0  # Floor across zero; never round to nearest.
    repeated = point_voxel_occupancy_metrics(torch.zeros(180000,3),torch.zeros(3,3))
    assert repeated['iou'] == 1 and repeated['predicted_occupied_voxels'] == 1
    for size in (0.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='Positive'):
            point_voxel_occupancy_metrics(torch.zeros(1,3),torch.zeros(1,3),voxel_size=size)


def test_density_runs_ignore_empty_and_roi():
    counts = torch.zeros(1,8,4,4)
    counts[0,2:6,2,2] = 1
    valid = torch.ones_like(counts,dtype=torch.bool)
    valid[:,4:] = False
    run = occupied_run_lengths(counts,valid)
    assert run[0,0,2,2,2] == .25
    assert run[0,1,3,2,2] == .25
    assert not run[:,:,4:].any()


def test_velocity_sign_endpoints_and_euler():
    source = torch.randn(180000,3)
    delta = torch.tensor([1.,2.,3.]).expand_as(source)
    endpoint = source+delta
    p0,v = flow_matching_pair(source,endpoint,torch.tensor([0.]))
    p1,_ = flow_matching_pair(source,endpoint,torch.tensor([1.]))
    assert torch.equal(source,p0) and torch.equal(endpoint,p1)
    assert torch.allclose(v,delta,atol=1e-6)
    ids = torch.zeros(180000,dtype=torch.long)
    assert torch.equal(euler_rollout(lambda p,t,i:torch.zeros_like(p),source,ids),source)
    for steps in (4,8,16):
        actual = euler_rollout(lambda p,t,i:delta,source,ids,steps=steps)
        assert torch.allclose(actual,endpoint,atol=1e-5)


def test_split_no_temporal_neighbors_or_seq08():
    frames = [{"sequence":"00","frame":i} for i in range(1000)]
    split = development_split(frames)
    assert len(split["development"]) == 100 and len(split["excluded_guard"]) == 40
    assert min(abs(a['frame']-b['frame']) for a in split['train'] for b in split['development']) > 20
    with pytest.raises(ValueError,match="seq08"):
        development_split([{"sequence":"08","frame":550}])


def test_labels_never_enter_loader_or_inference(tmp_path):
    import numpy as np
    p = np.array([[1.,2.,3.,99.]],dtype=np.float32)
    np.save(tmp_path/'input.npy',p)
    assert load_xyz(tmp_path/'input.npy').shape == (1,3)
    p[:,3] = float('nan')
    np.save(tmp_path/'input.npy',p)
    assert torch.equal(load_xyz(tmp_path/'input.npy'),torch.tensor([[1.,2.,3.]]))
    assert list(inspect.signature(pipeline.VoxelPointPipeline.__call__).parameters) == ['self','input_xyz','seed']
    tree = ast.parse(inspect.getsource(pipeline))
    imports = [getattr(node,'module','') for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any('teacher' in name or 'sinkhorn' in name or 'dataset' in name for name in imports)


def test_sparse_sinkhorn_no_dense_and_policy_labels(monkeypatch):
    from fpsgen.voxel_point import sinkhorn_refiner as refiner
    from fpsgen.ops.endpoint_refinement.sinkhorn import sparse_sinkhorn
    from fpsgen.ops.endpoint_refinement.sparse_global_ot import _sparse_knn_graph
    endpoint = torch.randn(64,3)
    target = endpoint+.02
    src,tgt,cost = _sparse_knn_graph(endpoint,target,4,'scipy')
    assert torch.unique(src).numel() == 64 and torch.unique(tgt).numel() == 64
    assert len(cost) <= 64*8
    plan = sparse_sinkhorn(src,tgt,cost,64,64,.05,200)
    assert torch.isfinite(plan).all()
    teacher_metrics = {'cd':1.,'coverage_0.1':.5,'spacing_mean':1.,'spacing_cv':1.,'out_of_range_ratio':0.}
    better = {**teacher_metrics,'cd':.5}
    monkeypatch.setattr(refiner,'point_metrics',lambda p,g,**kw:better if torch.equal(p,target) else teacher_metrics)
    def fake_plan(*args,**kwargs):
        return target,{'row_error':.01,'col_error':.01,'entropy':1.}
    monkeypatch.setattr(refiner,'refine_endpoint',fake_plan)
    strict,d = refiner.refine_with_policy(endpoint,target,small_spec(),mode='strict_balanced_ot',backend='scipy')
    assert torch.equal(strict,endpoint) and d['fallback_used'] and not d['balanced_ot_converged']
    guided,d = refiner.refine_with_policy(endpoint,target,small_spec(),mode='sinkhorn_guided_geometry',backend='scipy')
    assert torch.equal(guided,target) and d['geometry_refinement_accepted'] and not d['balanced_ot_converged']
    monkeypatch.setattr(refiner,'refine_endpoint',lambda *a,**k:(target,{'row_error':0.,'col_error':0.,'entropy':1.,'edge_plan_finite':True,'plan_every_source_target_covered':True}))
    accepted,d = refiner.refine_with_policy(endpoint,target,small_spec(),mode='strict_balanced_ot',backend='scipy')
    assert torch.equal(accepted,target) and d['balanced_ot_converged'] and not d['fallback_used']
    monkeypatch.setattr(refiner,'refine_endpoint',lambda *a,**k:(torch.full_like(target,float('nan')),{'row_error':float('nan'),'col_error':0.,'entropy':0.}))
    fallback,d = refiner.refine_with_policy(endpoint,target,small_spec(),backend='scipy')
    assert torch.equal(fallback,endpoint) and not d['plan_finite']


def test_source_provenance_files_match():
    import hashlib,json
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'third_party/voxel_point/SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        if 'adapted_path' in row:
            assert hashlib.sha256((root/row['adapted_path']).read_bytes()).hexdigest() == row['adapted_sha256']


def test_density_output_full_shape_fp32_softplus():
    from fpsgen.voxel_point.density_head import nonnegative_density
    spec = VoxelGridSpec()
    raw = torch.full((1,1,*spec.shape),-6.,dtype=torch.float16)
    density = nonnegative_density(raw,spec.valid_circle_mask()[None,None])
    assert density.shape == (1,1,256,256,32)
    assert density.dtype == torch.float32 and (density>=0).all() and torch.isfinite(density).all()


def test_grid_sample_coordinates_preserve_asymmetric_xyz():
    # Import this function's AST only so the coordinate contract does not
    # depend on optional upstream framework/spconv imports in CPU tests.
    source = Path(__file__).resolve().parents[1]/'fpsgen/voxel_point/vendor/voxdet/det_head.py'
    function = next(node for node in ast.parse(source.read_text()).body if getattr(node,'name',None)=='create_grid')
    namespace = {'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    x,y,z = torch.meshgrid(torch.arange(8),torch.arange(6),torch.arange(4),indexing='ij')
    feature = (100*x+10*y+z).float()[None,None]
    grid = namespace['create_grid'](x.float()[None],y.float()[None],z.float()[None],8,6,4,'cpu')
    reconstructed = torch.nn.functional.grid_sample(feature,grid,align_corners=False)
    assert torch.allclose(reconstructed,feature,atol=1e-4)


def test_teacher_contract_flags_and_source_indexing(monkeypatch):
    from fpsgen.voxel_point import mapping_teacher

    class FakePTv3(torch.nn.Module):
        def forward_conditional(self, points, centers, time, *, known_flags):
            assert points.shape == (360000,3)
            assert not known_flags[:180000].any() and known_flags[180000:].all()
            assert centers.shape == points.shape and time.item() == 0
            return torch.arange(360000).float()[:,None].expand(-1,3) * 1e-6

    monkeypatch.setattr(mapping_teacher,'build_ptv3',lambda *a,**k:FakePTv3())
    spec=VoxelGridSpec()
    gt=torch.zeros(180000,3)
    ids,_=spec.voxelize_points(gt)
    source=torch.ones_like(gt)*.01
    teacher=mapping_teacher.MappingTeacher(spec)
    endpoint=teacher(source,ids,gt)
    assert endpoint.shape == (180000,3)
    assert torch.allclose(endpoint,source+torch.arange(180000).float()[:,None].expand(-1,3)*1e-6)


def test_xyz_only_pipeline_preserves_180k_without_gt(tmp_path):
    spec=small_spec()

    class Density(torch.nn.Module):
        def forward(self,xyz):
            assert xyz.shape[1] == 3
            density=torch.zeros(1,1,*spec.shape)
            density[0,0,8,8,4]=1.
            return {'pred_density':density}

    class ZeroFlow(torch.nn.Module):
        def forward(self,points,time,voxel_ids):
            assert points.shape == (180000,3) and voxel_ids.shape == (180000,)
            return torch.zeros_like(points)

    model=pipeline.VoxelPointPipeline(Density(),ZeroFlow(),spec)
    a=model(torch.zeros(18,3),seed=17)
    b=model(torch.zeros(18,3),seed=17)
    assert torch.equal(a['anchors'],b['anchors']) and torch.equal(a['final_points'],a['anchors'])
    pipeline.save_inference(tmp_path/'scene',a,seed=17,spec=spec,provenance={'test_only':True})
    import numpy as np,json
    assert np.load(tmp_path/'scene/final_points.npy').shape == (180000,3)
    metadata=json.loads((tmp_path/'scene/manifest.json').read_text())
    assert metadata['point_count']==180000 and metadata['stage1_network_evaluations']==1


def test_official_voxdet_constructor_forward_bodies_preserved():
    root=Path(__file__).resolve().parents[1]
    references={
        'lidar_encoder.py':'models/backbones/lidar_encoder.py',
        'resnet3d.py':'models/backbones/resnet3d.py',
        'generalizedfpn.py':'models/necks/generalizedfpn.py',
        'spatial_deocupled_fpn_dense.py':'models/necks/spatial_deocupled_fpn_dense.py',
        'det_head.py':'models/dense_heads/det_head.py',
        'occ_head.py':'models/dense_heads/occ_head.py'}
    for name,relative in references.items():
        original=ast.parse((root/'third_party/voxel_point/VoxDet'/relative).read_text())
        adapted=ast.parse((root/'fpsgen/voxel_point/vendor/voxdet'/name).read_text())
        classes={node.name:node for node in original.body if isinstance(node,ast.ClassDef)}
        for node in adapted.body:
            if not isinstance(node,ast.ClassDef):continue
            methods={m.name:m for m in classes[node.name].body if isinstance(m,ast.FunctionDef)}
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name in ('__init__','forward'):
                    assert ast.dump(method,include_attributes=False)==ast.dump(methods[method.name],include_attributes=False)


def test_sqrt_chamfer_metres_and_differentiable_without_dense():
    from fpsgen.voxel_point.metrics import chamfer_sqrt
    p=torch.tensor([[3.,4.,0.]],requires_grad=True)
    target=torch.zeros(1,3)
    loss=chamfer_sqrt(p,target,backend='scipy')
    assert loss.item()==pytest.approx(5.)
    loss.backward()
    assert torch.allclose(p.grad,torch.tensor([[.6,.8,0.]]))
