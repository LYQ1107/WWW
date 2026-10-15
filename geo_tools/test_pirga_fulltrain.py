import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from audit_pirga_fulltrain import classify, checked
from pirga_indexed_store import StoreWriter, StoreReader, sparse_matrix, dense_matrix, epoch_order, coverage, sha


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'store';self.w=StoreWriter(self.root)
        self.x=np.linspace(.001,.8,1152,dtype=np.float32)
        self.ref=self.w.add_feature(4,[10,0],1,1,self.x)
        self.q=self.w.add_feature(4,[11,0],2,2,self.x)
        self.q2=self.w.add_feature(4,[11,1],2,2,self.x)
        shape,entries=sparse_matrix([[.8,.7],[.79,.1]])
        self.ref2=self.w.add_feature(4,[10,1],1,1,self.x)
        self.c=dict(scene=4,call_id=0,stage='run_global_tracker_plus',query_keys=[[4,11,0],[4,11,1]],query_features=[self.q,self.q2],
                    reference_features=[self.ref,self.ref2],reference_tracks=[101,102],candidate_ids=[101,102],reference_block_sizes=[2],
                    matrix_shape=shape,matrix_entries=entries,thresholds=[.1,.1],accepted_ids=[102,101],completed_image_ids=[])
    def tearDown(self):
        try:self.w.bin.close();self.w.db.close()
        finally:self.tmp.cleanup()
    def test_physical_dedup_and_distinct_bank_version(self):
        self.assertEqual(self.ref,self.w.add_feature(4,[10,0],1,1,self.x))
        with self.assertRaises(ValueError):self.w.add_feature(4,[10,0],1,1,self.x+.1)
        v=self.w.add_feature(4,[10,0],1,1,self.x+.1,variant='actual_version_sha')
        self.assertNotEqual(v,self.ref)
    def test_full_joint_hungarian_changes_rowwise_preference(self):
        self.w.add_call(self.c)
        m=self.w.close({'actual_source_sha':'externally_pinned'},2)
        r=StoreReader(self.root,sha(self.root/'manifest.json'));self.assertEqual(list(r.calls())[0]['accepted_ids'],[102,101])
        self.assertEqual(m['features'],4);self.assertGreater(m['quantization']['max_absolute_error'],0)
        np.testing.assert_array_equal(r.feature(self.ref),self.x.astype(np.float16).astype(np.float32))
    def test_unpinned_and_corrupt_rejected(self):
        self.w.add_call(self.c);self.w.close({},2)
        with self.assertRaises(ValueError):StoreReader(self.root,'0'*64)
        pin=sha(self.root/'manifest.json')
        with open(self.root/'native_vectors.bin','r+b') as f:f.write(b'xx')
        with self.assertRaises(ValueError):StoreReader(self.root,pin)
    def test_label_field_and_candidate_reconstruction_rejected(self):
        for field,value in [('GT_person',7),('positive_tracks',[101])]:
            c=copy.deepcopy(self.c);c[field]=value
            with self.assertRaises(ValueError):self.w.add_call(c)
        c=copy.deepcopy(self.c);c['candidate_ids']=[101]
        with self.assertRaises(ValueError):self.w.add_call(c)
    def test_future_uncommitted_and_cross_scene_rejected(self):
        future=self.w.add_feature(4,[12,0],3,1,self.x);c=copy.deepcopy(self.c);c['reference_features'][0]=future
        with self.assertRaises(ValueError):self.w.add_call(c)
        same=self.w.add_feature(4,[13,0],2,1,self.x);c['reference_features'][0]=same
        with self.assertRaises(ValueError):self.w.add_call(c)
        c['completed_image_ids']=[13];self.w.add_call(c)
    def test_wrong_original_action_and_strict_threshold(self):
        c=copy.deepcopy(self.c);c['accepted_ids']=[101,102]
        with self.assertRaises(ValueError):self.w.add_call(c)
        c=copy.deepcopy(self.c);c['thresholds']=[float(np.float32(.79)),float(np.float32(.7))];c['accepted_ids']=[-1,-1];self.w.add_call(c)
    def test_full_1152_and_finite_precision(self):
        for x in [np.ones(1000,np.float32),np.zeros(1152,np.float32),np.full(1152,np.nan,np.float32),np.full(1152,1e10,np.float32)]:
            with self.assertRaises(ValueError):self.w.add_feature(4,[20,0],1,1,x)
    def test_sparse_exact_and_duplicate_edge_rejected(self):
        x=np.array([[0,.5],[.3,0]],np.float32);s,e=sparse_matrix(x);np.testing.assert_array_equal(dense_matrix(s,e),x)
        with self.assertRaises(ValueError):dense_matrix(s,e+e[:1])
    def test_empty_candidate_call_retained(self):
        c=copy.deepcopy(self.c);c.update(reference_features=[],reference_tracks=[],candidate_ids=[],reference_block_sizes=[],matrix_shape=[2,0],matrix_entries=[],thresholds=[],accepted_ids=[-1,-1]);self.w.add_call(c)

    def test_lazy_actor_cannot_observe_labels_and_clones_features(self):
        from pirga_fulltrain_dataset import NativeQueryDataset,load_supervision
        from cv_pil_v2_model import tensorize,training_eligible
        self.w.add_call(self.c);self.w.close({},2)
        ds=NativeQueryDataset(self.root,sha(self.root/'manifest.json'))
        actor=ds.actor([4,11,0]);self.assertNotIn('GT_person',actor)
        before=actor['query_feature'].clone();actor['query_feature'].zero_()
        self.assertTrue((ds.actor([4,11,0])['query_feature']==before).all())
        import gzip
        lp=Path(self.tmp.name)/'offline_labels.jsonl.gz'
        labels=[dict(event_key=[4,11,0],GT_person=7,candidate_persons=[7,7],actual_primary_state='correct_link'),
                dict(event_key=[4,11,1],GT_person=None,candidate_persons=[7,None],actual_primary_state='unknown_query')]
        with gzip.open(lp,'wt') as f:
            for label in labels:f.write(json.dumps(label)+'\n')
        sup=load_supervision(lp,sha(lp),ds)
        self.assertEqual(sup[(4,11,0)]['certified_target_indices'],[0,1])
        self.assertEqual(sup[(4,11,1)]['certified_target_indices'],[])
        self.assertEqual(sup[(4,11,1)]['candidate_known_mask'],[True,False])
        self.assertFalse(training_eligible(tensorize(ds.actor([4,11,1])),sup[(4,11,1)]))
        self.assertTrue((ds.actor([4,11,0])['query_feature']==before).all())
    def test_unapproved_resource_permit_rejected_before_model_import(self):
        from export_pirga_native_prefix import validate_permit
        p=Path(self.tmp.name)/'permit.json';p.write_text(json.dumps(dict(status='PROPOSED_NOT_AUTHORIZED')))
        with self.assertRaises(ValueError):validate_permit(p,sha(p),4,40)


class ScheduleAndLabelsTests(unittest.TestCase):
    def test_all_eligible_once_every_epoch_shared_determinism(self):
        by={4:[(4,10,i) for i in range(100)],24:[(24,20,i) for i in range(137)]};full=sum(by.values(),[])
        orders=[]
        for epoch in range(1,6):
            a=list(epoch_order(by,20261016,epoch));orders.append(a);self.assertEqual(a,list(epoch_order(by,20261016,epoch)));self.assertTrue(coverage(full,a)['full_epoch_pass'])
        self.assertNotEqual(orders[0],orders[1]);self.assertEqual(len(orders),5)
    def test_development_and_duplicates_rejected(self):
        for by in [{12:[(12,10,0)]},{4:[(4,10,0)]*2},{4:[(5,10,0)]}]:
            with self.assertRaises(ValueError):list(epoch_order(by,1,1))
        self.assertFalse(coverage([(4,1,1),(4,1,2)],[(4,1,1)]*2)['full_epoch_pass'])
    def test_unknown_never_certified_wrong_and_categories_overlap(self):
        e=dict(query_gt_identity=None,positive_tracks=[],observed_positive_including_impure=[],actual_primary_state='unknown_query',known_wrong=False)
        c=classify(e);self.assertTrue(c['E_uncertified']);self.assertFalse(c['B_accepted_wrong_identity']);self.assertFalse(c['D_no_pure_legal_positive'])
        e.update(query_gt_identity=7,observed_positive_including_impure=[101],actual_primary_state='ambiguous_reference')
        c=classify(e);self.assertTrue(c['C_correct_observation_only_impure']);self.assertTrue(c['D_no_pure_legal_positive']);self.assertTrue(c['E_uncertified'])
    def test_source_sha_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source';p.write_text('original');pin=sha(p);checked(p,pin);p.write_text('changed')
            with self.assertRaises(ValueError):checked(p,pin)

if __name__=='__main__':unittest.main()
