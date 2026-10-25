#!/usr/bin/env python3
"""Audit delivered V3 IDs posthoc; no oracle state enters inference."""
from worldmot.evaluation.world_object_diagnostics import main

if __name__=="__main__":
    main("temporal_identity")
