"""Persistent bounded continuation: wait for real predecessor train+eval evidence."""
import json
import os
from pathlib import Path


def predecessor_status(predecessors):
    rows=[]
    for item in predecessors:
        path=Path(item['status_file'])
        if not path.exists():rows.append(dict(name=item['name'],status='WAITING',reason='Missing completion evidence'));continue
        value=json.loads(path.read_text());status=value.get('status')
        if status==item['required_status']:
            proofs=[]
            for check in item.get('proofs',[]):
                p=Path(check['path'])
                if not p.exists():proofs.append(False);continue
                data=json.loads(p.read_text());proofs.append(all(data.get(k)==v for k,v in check['equals'].items()))
            rows.append(dict(name=item['name'],status='COMPLETE' if all(proofs) else 'BLOCKED',reason='Verified own train/eval artifacts' if all(proofs) else 'Supervisor status without matching artifact proof'))
        elif any(word in str(status) for word in ('FAIL','BLOCKED','STOPPED','PAUSED')):
            rows.append(dict(name=item['name'],status='BLOCKED',reason='Predecessor needs review',actual_status=status))
        else:rows.append(dict(name=item['name'],status='WAITING',actual_status=status))
    state='READY' if rows and all(r['status']=='COMPLETE' for r in rows) else 'BLOCKED_PREDECESSOR' if any(r['status']=='BLOCKED' for r in rows) else 'WAITING_PREDECESSOR_EVAL'
    return dict(status=state,predecessors=rows)


def project_jobs(physical,report,root):
    """Do not co-locate a second project job even under shared external admission."""
    found=[]
    for pid in report.get('gpus',[{}])[0].get('existing_compute_pids',[]):
        if pid==os.getpid():continue
        path=Path('/proc')/str(pid)/'cmdline'
        try:command=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (PermissionError,FileNotFoundError):continue
        if str(root) in command:found.append(dict(pid=pid,command=command))
    return found
