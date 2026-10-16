"""One live interactive-agent check. Not scheduled by the launcher or monitor.

Call repeatedly from the active execution; these observations are separate from
the independent background monitor and never substitute waiting time for data.
"""
from common import ROOT
import argparse
import json
from pathlib import Path
from monitor_stage1_occ import snapshot


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    row=snapshot(args.output);row['scope_of_check']='interactive_agent_observation'
    with (args.output/'agent_checks.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
    latest=row['latest'] or {}
    print(json.dumps({'epoch_time':row['epoch_time'],'successful_updates':row['successful_updates'],'attempts':row['attempts'],
        'loss':latest.get('loss'),'last100_loss':row['last100_loss_mean'],'gradient_norms':latest.get('gradient_norms'),
        'amp_scale':latest.get('amp_scale'),'overflows':latest.get('amp_overflow_count'),
        'parameters_changed':latest.get('parameter_changes'),'processes':row['processes'],'gpu':row['gpu'],
        'latest_update_age_seconds':row['latest_update_age_seconds'],'checkpoint_exists':row['checkpoint_exists'],
        'disk_free_bytes':row['disk_free_bytes']},indent=2))

if __name__=='__main__':main()
