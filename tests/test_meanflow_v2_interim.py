import copy
import pytest
from scripts.research_v2.meanflow_v2.interim_guards import validate_interim_checkpoint,validate_own_sharing
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha


def checkpoint():
    cfg={'arm':'MFV2-A','physical_gpu':2}
    return cfg,dict(identity=dict(initialization_mode='scratch',arm='MFV2-A',config_sha256=config_sha(cfg),
        source_sha256='training-source',geometry_sha256='original-geometry'),
        counts=dict(updates=7500,exposures=15000,attempts=7502,amp_skips=2,cursor=15004),scheduler={'last_epoch':7500})


def test_explicit_partial_budget_is_accepted_and_not_mislabelled():
    cfg,s=checkpoint();validate_interim_checkpoint(s,cfg,7500,'training-source','original-geometry')


@pytest.mark.parametrize('change',['final','exposures','source','counter','geometry','schedule'])
def test_rejects_wrong_identity_or_budget(change):
    cfg,s=checkpoint()
    if change=='final':s['counts'].update(updates=10000,exposures=20000)
    if change=='exposures':s['counts']['exposures']=20000
    if change=='source':s['identity']['source_sha256']='changed'
    if change=='counter':s['counts']['cursor']+=2
    if change=='geometry':s['identity']['geometry_sha256']='changed'
    if change=='schedule':s['scheduler']['last_epoch']+=1
    with pytest.raises(ValueError):validate_interim_checkpoint(s,cfg,7500,'training-source','original-geometry')


def test_shared_resource_guard_rejects_foreign_process_or_low_memory():
    report={'gpus':[dict(free_mib=12000,existing_compute_pids=[1351803])]}
    validate_own_sharing(report,1351803)
    q=copy.deepcopy(report);q['gpus'][0]['existing_compute_pids'].append(123)
    with pytest.raises(RuntimeError):validate_own_sharing(q,1351803)
    q=copy.deepcopy(report);q['gpus'][0]['free_mib']=7000
    with pytest.raises(RuntimeError):validate_own_sharing(q,1351803)
