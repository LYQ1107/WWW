"""Actual GMT relation adapter zero-contract gate; trained mode remains gated."""
import argparse
import torch
from common import *


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--zero-contract',action='store_true')
    args,remaining=parser.parse_known_args()
    if not args.zero_contract:raise ValueError('trained student deployment requires a validated teacher/student checkpoint; currently zero-contract only')
    import run_online
    from gtr.geometry.identity_relation_adapter import IdentityRelationAdapter
    original_load=run_online.load_gmt;original_report=run_online.provenance;original_argv=list(sys.argv);stats=dict(calls=0,exact_object_returns=0)
    def load(*args,**kwargs):
        cfg,model=original_load(*args,**kwargs)
        if model.geo_runtime.active:raise ValueError('zero-contract must pair with native GMT OFF')
        neutral=IdentityRelationAdapter();original=model.geo_runtime.adapt
        def adapt(outputs,instances,k):
            value=original(outputs,instances,k)
            refs=[instance for index,instance in enumerate(instances) if index!=k]
            if not refs:return value
            tracks=torch.cat([instance.track_ids for instance in refs]);unique=tracks.unique(sorted=True)
            for matrix in value:
                candidate=unique[None].expand(matrix.shape[0],-1)
                result=neutral(matrix,tracks,candidate,matrix.new_zeros(candidate.shape))
                stats['calls']+=1;stats['exact_object_returns']+=int(result is matrix)
                if result is not matrix:raise ValueError('zero residual changed original logit object')
            return value
        model.geo_runtime.adapt=adapt
        return cfg,model
    def report():
        value=original_report();value.update(command=original_argv,validation_relation_zero=True,
            relation_adapter=dict(stats),relation_student_checkpoint_loaded=False,relation_teacher_or_cache_loaded=False,
            relation_scope='real original causal GMT with explicit unique-track-to-proposal zero-residual adapter before unchanged GTA/Hungarian/memory; not trained-student efficacy')
        return value
    run_online.load_gmt=load;run_online.provenance=report
    try:sys.argv=[original_argv[0],*remaining];run_online.main()
    finally:sys.argv=original_argv;run_online.load_gmt=original_load;run_online.provenance=original_report


if __name__=='__main__':main()
