import tempfile,unittest
from pathlib import Path
from audit_view_order_runs import option_signature


class Tests(unittest.TestCase):
    def test_worktree_paths_require_identical_bytes_and_other_options(self):
        with tempfile.TemporaryDirectory() as directory:
            left,right=[Path(directory)/name for name in ['left.json','right.json']]
            left.write_text('{"slope":1}');right.write_text(left.read_text())
            a=dict(CALIBRATION_PATH=str(left),FEATURE_WEIGHT=2)
            b=dict(CALIBRATION_PATH=str(right),FEATURE_WEIGHT=2)
            self.assertEqual(option_signature(a),option_signature(b))
            self.assertNotEqual(option_signature(a),option_signature(dict(b,FEATURE_WEIGHT=1)))
            right.write_text('{"slope":2}')
            self.assertNotEqual(option_signature(a),option_signature(b))


if __name__=='__main__':unittest.main()
