"""Explicit user-directed numerical pilot; failed certification remains FAILED."""
import json
from pathlib import Path
from .storage import file_sha


def validate_direct_policy(path, cfg, report, source, config_digest):
    policy = json.loads(Path(path).read_text())
    if (policy.get('status') != 'USER_DIRECTED_NUMERICAL_PILOT'
            or policy.get('execution_source_sha256') != source
            or policy.get('formal_exposures') != 20000 or policy.get('batch') != 2
            or policy.get('seed') != 20261032 or cfg['arm'] not in ('EF1', 'EF2')):
        raise ValueError('Wrong direct-pilot policy/source/budget')
    evidence = policy['arms'][cfg['arm']]
    if (evidence['engineering_report_sha256'] != file_sha(cfg['engineering_report'])
            or evidence['config_sha256'] != config_digest
            or report['config_sha256'] != config_digest
            or report['source_sha256'] != policy['engineering_source_sha256']):
        raise ValueError('Direct pilot evidence changed')
    checks = report['checks']
    failures = {k for k, value in checks.items() if value is not True}
    if (report['status'] != 'FAIL'
            or failures != {'resume_parameter_bound', 'resume_auxiliary_bound'}
            or not all(checks.get(k) is True for k in (
                'required_gradients', 'frozen_state_exact', 'encoder_changed',
                'finite', 'resume_loss_exact', 'resume_rng_exact', 'resume_scalars_exact'))
            or report['true_batch'] != 2 or not report['no_token_truncation']
            or not report['stress_census_coverage']['complete']):
        raise ValueError('Only the disclosed numerical replay failure is excepted')
    # Exception limits only: these do not turn the original strict check into PASS.
    for key, limit in [('resume_parameter_differences', 2e-7),
                       ('resume_auxiliary_differences', 2e-6)]:
        rows = report[key]
        if not rows or any(not 0 <= r['max_abs'] <= limit for r in rows):
            raise ValueError('Numerical difference exceeds the disclosed pilot exception')
    if policy.get('verified_data_receipt'):
        if file_sha(policy['verified_data_receipt'])!=policy['verified_data_receipt_sha256']:
            raise ValueError('Full data audit receipt changed')
    return {'policy_sha256': file_sha(path), 'engineering_status': 'FAIL',
            'resume_certification': 'NOT_CERTIFIED_NUMERICAL_REPLAY',
            'user_instruction': policy['user_instruction'],
            'engineering_report_sha256': evidence['engineering_report_sha256'],
            'startup_health': 'in-run at32/100/500 updates; same optimizer/schedule',
            'verified_data_receipt':policy.get('verified_data_receipt'),
            'formal_exposures': 20000, 'batch': 2, 'seed': 20261032}
