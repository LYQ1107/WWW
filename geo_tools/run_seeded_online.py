"""Original causal entry with an explicit actual inference RNG seed."""
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--inference-seed',type=int,required=True)
    original_argv=sys.argv;command=list(original_argv);args,remaining=parser.parse_known_args()
    import run_online
    original_load=run_online.load_gmt;original_seed=run_online.seed_all;original_provenance=run_online.provenance
    def load(*values,**kwargs):
        kwargs['seed']=args.inference_seed
        return original_load(*values,**kwargs)
    def report():
        result=original_provenance(seed=args.inference_seed)
        result.update(command=command,explicit_inference_seed=args.inference_seed,
            inference_seed_scope='model construction and pre-stream reset both use this actual RNG seed; frozen weights/head unchanged; TRAIN-head seed is separate')
        return result
    run_online.load_gmt=load;run_online.seed_all=lambda ignored:original_seed(args.inference_seed);run_online.provenance=report
    try:
        sys.argv=[command[0],*remaining];run_online.main()
    finally:
        sys.argv=original_argv;run_online.load_gmt=original_load;run_online.seed_all=original_seed;run_online.provenance=original_provenance


if __name__=='__main__':main()
