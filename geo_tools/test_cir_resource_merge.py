"""Cost imports preserve old evidence and cannot reset or count costs twice."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from merge_cir_subledger import main


class CostMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name)/'parent.json';self.child=Path(self.temp.name)/'child.json'
        self.base=dict(allocated_total_GPU_wall_hours=2.,measured_GPU_wall_hours=1.,
            stages=[dict(stage='historic',status='COMPLETE',conservative_GPU_wall_hours=1.)])
        self.leaf=dict(parent_root=str(self.parent),parent_existing_credit_only=True,new_GPU_credit=0,
            allocated_total_GPU_wall_hours=.5,measured_GPU_wall_hours=.2,
            stages=[dict(stage='next',status='COMPLETE',conservative_GPU_wall_hours=.2)])
        self.write()
    def write(self):
        self.parent.write_text(json.dumps(self.base));self.child.write_text(json.dumps(self.leaf))
    def merge(self):
        with patch.object(sys,'argv',['merge','--parent',str(self.parent),'--child',str(self.child)]):main()
    def test_actual_cost_added_once_and_child_bytes_unchanged(self):
        before=self.child.read_bytes();self.merge();p=json.loads(self.parent.read_text())
        self.assertEqual(p['measured_GPU_wall_hours'],1.2);self.assertEqual(p['allocated_total_GPU_wall_hours'],2.)
        self.assertEqual(p['stages'][0],self.base['stages'][0]);self.assertEqual(self.child.read_bytes(),before)
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual(json.loads(self.parent.read_text())['measured_GPU_wall_hours'],1.2)
    def test_live_writer_blocks_import_without_parent_mutation(self):
        self.leaf['stages'][0]['status']='RUNNING';self.write();before=self.parent.read_bytes()
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual(before,self.parent.read_bytes())
    def test_cost_mismatch_cannot_underreport(self):
        self.leaf['measured_GPU_wall_hours']=0.;self.write()
        with self.assertRaises(ValueError):self.merge()
    def test_existing_credit_cannot_be_silently_increased(self):
        self.leaf['new_GPU_credit']=1.;self.write()
        with self.assertRaises(ValueError):self.merge()
    def test_overspend_is_rejected(self):
        self.base['allocated_total_GPU_wall_hours']=1.1;self.write()
        with self.assertRaises(ValueError):self.merge()


if __name__=='__main__':unittest.main()
