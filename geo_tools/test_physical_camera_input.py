import unittest
import common
from gtr.geometry.physical_camera_input import PhysicalCameraGMT


class Tracker:
    def __init__(self,model):self.calls=[]
    def step(self,records):
        self.calls.append(records)
        return [dict(view=record['view_id'],record=record) for record in records]


class Tests(unittest.TestCase):
    def test_canonical_input_objects_and_caller_output_positions(self):
        adapter=PhysicalCameraGMT(None,Tracker);a=dict(view_id=7);b=dict(view_id=2)
        result=adapter.step([a,b]);self.assertIs(adapter.calls[0][0],b);self.assertIs(adapter.calls[0][1],a)
        self.assertIs(result[0]['record'],a);self.assertIs(result[1]['record'],b)
        adapter.step([b,a]);self.assertEqual(adapter.camera_ids,(2,7));self.assertIs(adapter.calls[1][0],b)

    def test_reject_ambiguous_or_changed_membership_before_tracker(self):
        for bad in ([],[{}],[dict(view_id=True)],[dict(view_id=1.),dict(view_id=2)],[dict(view_id=1),dict(view_id=1)]):
            adapter=PhysicalCameraGMT(None,Tracker)
            with self.assertRaises(ValueError):adapter.step(bad)
            self.assertFalse(adapter.calls)
        adapter=PhysicalCameraGMT(None,Tracker);adapter.step([dict(view_id=1),dict(view_id=2)])
        with self.assertRaises(ValueError):adapter.step([dict(view_id=1),dict(view_id=3)])
        self.assertEqual(len(adapter.calls),1)


if __name__=='__main__':unittest.main()
