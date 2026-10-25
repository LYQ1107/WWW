#!/usr/bin/env python3
"""Retired V1 queue: cannot launch workers, even with a later V3 approval."""
import argparse
from worldmot.runtime.experiment_authorization import require_experiment_authorization


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorization");p.add_argument("--gpu-ids",nargs="+",required=True)
    p.add_argument("--phase",required=True);p.add_argument("--root",required=True)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    require_experiment_authorization(a.authorization,"schedule",method_id="FULL_WORLD_OBJECT_V3")
    raise RuntimeError("LEGACY_SIX_GPU_QUEUE_DISABLED_V3: use approved phase commands")


if __name__=="__main__":
    main()
