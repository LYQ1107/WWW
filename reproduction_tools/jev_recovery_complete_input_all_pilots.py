"""Separate final all-pilot audit; preserve the earlier available-pilot result."""
from jev_recovery_common import REPORTS
import jev_recovery_complete_input_assessment as audit

if __name__=='__main__':
    audit.REPORTS=REPORTS/'complete_input_all_pilots_v2'
    audit.main()
