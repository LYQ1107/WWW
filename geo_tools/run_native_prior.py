"""Explicit fixed-prior control or paired baseline with an actual inference seed."""
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--native-prior',action='store_true')
    parser.add_argument('--inference-seed',type=int,default=20261014)
    original_command=list(sys.argv);options,remaining=parser.parse_known_args()
    import run_online
    loaded=[]
    original_load=run_online.load_gmt;original_seed=run_online.seed_all;original_provenance=run_online.provenance
    def load(*args,**kwargs):
        kwargs['seed']=options.inference_seed
        cfg,model=original_load(*args,**kwargs)
        if options.native_prior:
            from gtr.geometry.native_prior import install
            install(model.geo_runtime)
        elif model.geo_runtime.active:raise ValueError('paired baseline must have GEO inactive')
        loaded.append(model)
        return cfg,model
    def report():
        result=original_provenance(seed=options.inference_seed)
        result.update(command=original_command,validation_native_output_prior=options.native_prior,
            native_prior_scope='fixed+1 evidence for finite native-GMT observations in other cameras within20 observed groups, multiplied/clipped by configured pre-activation adapter; no head/foundation vectors or new learned parameters' if options.native_prior else 'paired causal GMT baseline with explicit inference RNG seed',
            native_prior_calibration_scope='constructor parses ordinary native calibration metadata, but fixed-prior evidence does not use its fitted values' if options.native_prior else None)
        if options.native_prior:
            runtime=loaded[0].geo_runtime
            if runtime.vggt is not None or runtime.head is not None or runtime.head_checkpoint is not None:
                raise ValueError('head-free prior unexpectedly loaded an extra model')
            result.update(native_prior_extra_learned_parameters=0,native_prior_extra_encoder_or_head_loaded=False,
                          native_prior_observation_stats=dict(runtime.native_prior_stats))
        return result
    run_online.load_gmt=load;run_online.seed_all=lambda ignored:original_seed(options.inference_seed);run_online.provenance=report
    try:
        sys.argv=[original_command[0],*remaining]
        run_online.main()
    finally:
        sys.argv=original_command;run_online.load_gmt=original_load;run_online.seed_all=original_seed;run_online.provenance=original_provenance


if __name__=='__main__':main()
