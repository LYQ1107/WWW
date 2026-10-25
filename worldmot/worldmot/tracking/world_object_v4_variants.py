"""Finite registered V4 interventions. No TEST-dependent parameter search."""
from dataclasses import replace
from itertools import product
from .world_object_config import WorldObjectConfig


def factorial_configs(feature_dim=1024):
    base=WorldObjectConfig(feature_dim)
    return {f'G1_{int(a)}_G2_{int(b)}_G3_{int(c)}':replace(base,
        enable_sync_geometry=a,enable_temporal_geometry=b,enable_spatial_state_update=c)
        for a,b,c in product((False,True),repeat=3)} | {
        'G4_OFF':replace(base,enable_head_geometry=False)}


def state_configs(feature_dim=1024):
    c0=WorldObjectConfig(feature_dim)
    c1=replace(c0,enable_motion_prediction=True)
    c2=replace(c1,geometry_policy='UNCERTAINTY_SOFT')
    c3=replace(c2,enable_safe_identity_memory=True)
    c4=replace(c3,enable_multiview_coordination=True)
    return dict(C0=c0,C1=c1,C2=c2,C3=c3,C4=c4,C5=replace(c4,enable_omega_scheduler=True))
