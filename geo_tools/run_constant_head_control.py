"""Validation-only score ablation; actual feature extraction remains charged."""
import sys
from common import *
from gtr.geometry.lightweight_head import LowRankIdentityHead


def main():
    if '--head-path' not in sys.argv:raise ValueError('explicit frozen head provenance required')
    import run_online
    original=LowRankIdentityHead.evidence
    original_provenance=run_online.provenance
    def constant(self,query,reference):
        return query.new_ones((len(query),len(reference)))
    def control_provenance():
        report=original_provenance()
        report.update(validation_constant_head_evidence=1.,
            constant_control_scope='learned pair evidence replaced by the fixed upper-bound+1; original feature encoders, projection, feature validity, temporal/camera masks and bounded score adapter retained; feature weight2 gives at most+2 before native per-frame activation; no DEV/TEST fit',
            constant_control_parameter_scope='frozen head is still loaded and encodes targets for matched eligibility; its learned pair evidence is unused, so this is a score-discrimination ablation, not a zero-parameter production implementation')
        return report
    LowRankIdentityHead.evidence=constant;run_online.provenance=control_provenance
    try:
        if '--trace-association' not in sys.argv:sys.argv.append('--trace-association')
        run_online.main()
    finally:
        LowRankIdentityHead.evidence=original;run_online.provenance=original_provenance


if __name__=='__main__':main()
