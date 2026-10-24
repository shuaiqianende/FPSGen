"""P3 engineering-only selective E2 adapters. No training entry point."""
from .fidelity_dense_blocks import replace_fidelity_stage0


def prepare_selective_stage0(encoder,target,*,rank=64):
    if target not in ('ER-Local','ER-Voxel'):raise ValueError('Only registered selective E2 designs')
    if encoder.encoder_variant!='original':raise ValueError('Start from an independent original scratch Encoder')
    chosen='local_backbone' if target=='ER-Local' else 'voxel_backbone'
    count=replace_fidelity_stage0(getattr(encoder,chosen),'asymmetric3d',asymmetric_rank=rank)
    expected=3 if target=='ER-Local' else 2
    if count!=expected:raise ValueError('Unexpected original Stage0 block count')
    encoder.encoder_variant='asymmetric3d'
    # The existing forward adapter leaves the other original backbone intact.
    encoder.omd_replaced_blocks={'voxel_backbone':count if chosen=='voxel_backbone' else 0,
        'local_backbone':count if chosen=='local_backbone' else 0}
    encoder._p3_engineering_only=True
    return {'candidate':target,'rank':rank,'replaced_blocks':encoder.omd_replaced_blocks,
            'formal_training_go':False,'historical_scratch_seed_mismatch_requires_control':True}


def build_scratch_engineering_model(geometry,target):
    import torch
    from .stage1_occupancy import Stage1DirectOccupancy
    # Construct the entire original model before replacement so the original
    # Flow and all untouched Encoder parameters share their initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261032)
        model=Stage1DirectOccupancy(geometry,objective='fm',widths=(8,16,32,64),activation_checkpointing=True)
        audit=prepare_selective_stage0(model.encoder,target)
    return model,audit
