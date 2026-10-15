import copy
import unittest
from preflight import validate_locked_split


class Tests(unittest.TestCase):
    def fixture(self):
        split=dict(association_train_video_ids=[1],full_development_video_ids=[2,3],strict_development_video_ids=[2],raw_gt_diagnostic_development_video_ids=[3],minimal_development_video_ids=[2],official_test_used_for_this_project=False,historical_test_exposure=True)
        lock=dict(status='LOCKED_FULL_STRICT_DEVELOPMENT_BASELINE',split=copy.deepcopy(split),strict_video_runs={'2':{}},test_used_for_current_selection=False,historical_test_exposure=True)
        return lock,split

    def test_validation_does_not_modify_lock_or_split(self):
        lock,split=self.fixture();before=copy.deepcopy((lock,split))
        self.assertEqual(validate_locked_split(lock,split,[1,2,3]),[2]);self.assertEqual((lock,split),before)

    def test_partition_scene_and_exposure_changes_rejected(self):
        for change in [dict(association_train_video_ids=[1,2]),dict(strict_development_video_ids=[2,3]),dict(strict_development_video_ids=[2,2]),dict(official_test_used_for_this_project=True),dict(historical_test_exposure=False)]:
            lock,split=self.fixture();split.update(change);lock['split']=copy.deepcopy(split)
            with self.assertRaises(ValueError):validate_locked_split(lock,split,[1,2,3])
        lock,split=self.fixture();lock['split']={}
        with self.assertRaises(ValueError):validate_locked_split(lock,split,[1,2,3])


if __name__=='__main__':unittest.main()
