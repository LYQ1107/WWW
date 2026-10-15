import unittest
import torch
from capture_trackvggt_selected_v3 import ObservationWindow, selected_call, retain_observations


def observation(frame,camera=1):
    return dict(frame=frame,camera=camera,feature=torch.ones(3,dtype=torch.float16))


class SelectedCaptureTests(unittest.TestCase):
    def test_window_evicts_unselected_features_without_mutating_retained(self):
        w=ObservationWindow();w.add((1,0),observation(1));w.add((2,0),observation(2));kept={(1,0):w.values[1,0]}
        w.evict(2);self.assertEqual(set(w.values),{(2,0)});self.assertEqual(kept[1,0]['frame'],1)
        with self.assertRaises(ValueError):w.add((2,0),observation(2))

    def test_selected_rows_native_only_and_latest_four(self):
        w=ObservationWindow();sources=[(i,0) for i in range(1,7)]
        for i,key in enumerate(sources):w.add(key,observation(i+1,camera=1+i%2))
        w.add((7,0),observation(7));w.add((7,1),observation(7))
        scores=torch.tensor([[1.,2.],[4.,3.]]);logits=torch.arange(12).float().reshape(2,6)
        c,events=selected_call(scores,logits,[(7,0),(7,1)],sources,torch.tensor([10]*5+[20]),torch.tensor([10,20]),[3,3],w,{10:99,20:1},{(7,1)})
        self.assertEqual(len(events),1);self.assertEqual(events[0]['query_row'],0);self.assertEqual(events[0]['original_query_row'],1)
        self.assertTrue(torch.equal(c['logits'][0],logits[1]));self.assertEqual(events[0]['candidate_tracks'],[10,20])
        self.assertEqual(c['histories'][10],dict(sources=[[2,0],[3,0],[4,0],[5,0]],observed_length=99))
        kept={};retain_observations(w,c,events[0],kept)
        self.assertNotIn((1,0),kept);self.assertNotIn((7,0),kept);self.assertIn((7,1),kept)

    def test_future_history_rejected(self):
        w=ObservationWindow();w.add((1,0),observation(1));w.add((2,0),observation(2))
        with self.assertRaises(ValueError):retain_observations(w,dict(histories={10:dict(sources=[[2,0]])}),dict(query_source=[1,0],candidate_tracks=[10]),{})

    def test_empty_candidates_are_observed_empty_not_fake_features(self):
        w=ObservationWindow();w.add((1,0),observation(1))
        c,events=selected_call(torch.empty(1,0),torch.empty(1,0),[(1,0)],[],torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),[0],w,{}, {(1,0)})
        self.assertEqual(events[0]['candidate_tracks'],[]);self.assertEqual(c['histories'],{})
        kept={};retain_observations(w,c,events[0],kept);self.assertEqual(set(kept),{(1,0)})


if __name__=='__main__':unittest.main()
