import unittest
from probe_live_camera_order import partition


class Tests(unittest.TestCase):
    def test_one_global_renaming_is_equivalent(self):
        rows=lambda ids:[dict(track_id=identity) for identity in ids]
        self.assertEqual(partition(rows([1,2,1,3,2])),partition(rows([9,7,9,4,7])))

    def test_merge_split_and_time_dependent_renaming_are_not_equivalent(self):
        rows=lambda ids:[dict(track_id=identity) for identity in ids]
        reference=partition(rows([1,2,1,2]))
        for changed in ([9,9,9,9],[9,7,4,7],[9,7,7,9]):
            self.assertNotEqual(reference,partition(rows(changed)))


if __name__=='__main__':unittest.main()
