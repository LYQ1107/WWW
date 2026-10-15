"""Run original causal entry with actual-assignment provenance, no GT labels."""
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--tag',required=True)
    args,_=parser.parse_known_args();out=RUNTIME/'online'/args.tag
    from gtr.geometry.decision_trace import DecisionTrace
    import run_online
    traces=[];original=run_online.load_gmt
    def load(*args,**kwargs):
        cfg,model=original(*args,**kwargs);trace=DecisionTrace(model,out);trace.install();traces.append(trace)
        return cfg,model
    run_online.load_gmt=load
    # Existing trace flag explicitly excludes this run from speed tables.
    if '--trace-association' not in sys.argv:sys.argv.append('--trace-association')
    complete=False
    try:run_online.main();complete=True
    finally:
        for trace in traces:trace.close()
        if traces:
            names=['actual_decisions.jsonl.gz','detector_provenance.jsonl.gz']
            save_json(out/'decision_trace_manifest.json',dict(status='COMPLETE' if complete else 'FAILED',
                speed_eligible=False,images=len(traces[0].seen),image_ids=sorted(traces[0].seen),actual_tracker_calls=traces[0].counts,
                artifacts={name:sha256(out/name) for name in names},provenance=provenance(),
                scope='actual original solver returns retained; no GT enters inference; diagnostic source tags survive slices/cat; canonical standalone prediction parity is a separate required gate'))


if __name__=='__main__':main()
