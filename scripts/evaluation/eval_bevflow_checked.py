#!/usr/bin/env python3
"""Run the unchanged BEV evaluator, recording context if sampling fails.

No retry, clipping, precision fallback, or frame skipping is performed.
Successful sampling calls execute the original sampler without extra hooks.
"""
import inspect
import json
import sys
from pathlib import Path

from scripts import eval_bevflow_lidar_only as evaluator
from scripts.evaluation.diagnose_bev_sampling_nonfinite import stats, trace


def main():
    compatibility_fix = '--inductor-beta0-fix' in sys.argv
    if compatibility_fix:
        sys.argv.remove('--inductor-beta0-fix')
        from scripts.evaluation.inductor_beta0_fix import install
        install()
        print('RUNTIME FIX: Inductor baddbmm beta=0 ignores additive input; FP16 unchanged',flush=True)
    original = evaluator.sample_bev_cfg

    def checked(flow, raw, layout, **options):
        recorded = []
        # Retain existing output tensors, but do not sync/copy or alter them.
        # This permits inspection of the actual failed call, not just replay.
        dense = flow._eval_dense_forward
        def observed(*inputs):
            velocity = dense(*inputs)
            recorded.append((inputs[0], inputs[1], velocity))
            return velocity
        flow._eval_dense_forward = observed
        try:
            return original(flow, raw, layout, **options)
        except FloatingPointError:
            caller = inspect.currentframe().f_back
            frame = caller.f_locals.get('stem')
            output = caller.f_locals.get('output')
            failure = dict(frame=frame, sampling=options,
                           inductor_beta0_fix=compatibility_fix,
                           raw_condition=stats(raw), layout=stats(layout),
                           original_steps=[dict(step=i, input_state=stats(x),
                                                time=stats(t), velocity=stats(v))
                                           for i,(x,t,v) in enumerate(recorded)])
            flow._eval_dense_forward = dense
            print('SAMPLING FAILURE', json.dumps(failure), flush=True)
            # These are forensic replays, never substitutes for the failed
            # evaluation sample or for headline metrics.
            for label, dense, precision, hooks in [
                ('fp16_runtime_replay', flow._eval_dense_forward, 'fp16', False),
                ('fp16_eager', flow.model.forward, 'fp16', True),
                ('fp32_eager', flow.model.forward, 'fp32', True),
            ]:
                try:
                    failure[label] = trace(
                        flow, raw, dense=dense, precision=precision,
                        cfg=options['guidance_scale'], hooks=hooks,
                        seed=options['seed'], steps=options['steps'], layout=layout,
                    )
                except Exception as exc:
                    failure[label] = dict(error=repr(exc))
            if output is not None:
                path = Path(output) / 'sampling_failure.json'
                path.write_text(json.dumps(failure, indent=2) + '\n')
                print('FAILURE DETAILS', str(path), flush=True)
            raise
        finally:
            flow._eval_dense_forward = dense

    evaluator.sample_bev_cfg = checked
    evaluator.main()
    if compatibility_fix:
        output = Path(sys.argv[sys.argv.index('--output') + 1])
        summary_path = output / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['runtime_fixes'] = ['pytorch20_inductor_baddbmm_beta0']
        summary_path.write_text(json.dumps(summary,indent=2) + '\n')


if __name__ == '__main__':
    main()
