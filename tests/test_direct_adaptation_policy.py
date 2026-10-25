import copy
import json
import pytest
from fpsgen.hier_occ_flow.storage import file_sha
from fpsgen.hier_occ_flow.direct_adaptation_policy import validate_direct_policy


def evidence(tmp_path):
    checks={k:True for k in ('required_gradients','frozen_state_exact','encoder_changed','finite',
                            'resume_loss_exact','resume_rng_exact','resume_scalars_exact')}
    checks.update(resume_parameter_bound=False,resume_auxiliary_bound=False)
    report=dict(status='FAIL',checks=checks,true_batch=2,no_token_truncation=True,
                stress_census_coverage={'complete':True},source_sha256='old',config_sha256='cfg',
                resume_parameter_differences=[dict(max_abs=1.4e-7)],
                resume_auxiliary_differences=[dict(max_abs=1e-6)])
    path=tmp_path/'evidence.json';path.write_text(json.dumps(report))
    cfg=dict(arm='EF1',engineering_report=str(path))
    policy=dict(status='USER_DIRECTED_NUMERICAL_PILOT',execution_source_sha256='new',
                engineering_source_sha256='old',formal_exposures=20000,batch=2,seed=20261032,
                user_instruction='开始正式训练啊',arms={'EF1':dict(engineering_report_sha256=file_sha(path),config_sha256='cfg')})
    file=tmp_path/'policy.json';file.write_text(json.dumps(policy))
    return cfg,report,policy,file,path


def test_exception_never_relabels_failed_replay(tmp_path):
    cfg,report,policy,file,path=evidence(tmp_path)
    result=validate_direct_policy(file,cfg,report,'new','cfg')
    assert report['status']=='FAIL' and result['engineering_status']=='FAIL'
    assert result['resume_certification']=='NOT_CERTIFIED_NUMERICAL_REPLAY'


@pytest.mark.parametrize('key',['frozen_state_exact','finite','required_gradients','resume_rng_exact'])
def test_other_failures_cannot_be_excepted(tmp_path,key):
    cfg,report,policy,file,path=evidence(tmp_path)
    report['checks'][key]=False;path.write_text(json.dumps(report))
    policy['arms']['EF1']['engineering_report_sha256']=file_sha(path);file.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match='Only the disclosed'):validate_direct_policy(file,cfg,report,'new','cfg')


def test_changed_evidence_or_source_is_rejected(tmp_path):
    cfg,report,policy,file,path=evidence(tmp_path)
    with pytest.raises(ValueError):validate_direct_policy(file,cfg,report,'different','cfg')
    path.write_text('{}')
    with pytest.raises(ValueError):validate_direct_policy(file,cfg,report,'new','cfg')
