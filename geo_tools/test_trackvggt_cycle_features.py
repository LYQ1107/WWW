import unittest
import numpy as np
import torch
from types import SimpleNamespace
from trackvggt_cycle_features import relation_inputs,forward_cycle


class CycleFeatureTests(unittest.TestCase):
    def test_missing_native_pair_and_future_are_rejected_not_filled(self):
        arrays={f'layer_{k}':np.ones((2,4),np.float16) for k in (4,11,17,23)}
        arrays.update(attention=np.ones(2),xy=np.zeros((2,2)),inside=np.ones(2,bool),roi=np.ones((4,4)))
        data=SimpleNamespace(arrays=lambda key:arrays)
        query=dict(source=[3,0],feature=torch.ones(8),frame=3,camera=42)
        obs={(2,0):dict(source=[2,0],feature=torch.ones(8),frame=2,camera=7)}
        nodes=[dict(track=9,history_sources=[[2,0]],observed_length=10)]
        with self.assertRaises(ValueError):relation_inputs(data,4,query,nodes,[None],1.,.1,obs)
        result=relation_inputs(data,4,query,nodes,[.5],1.,.1,obs)
        self.assertEqual(result['pair'].shape,(1,8));self.assertEqual(result['histories'][0]['observed_length'],10)
        obs[(2,0)]['GT_identity']=999;query['GT_identity']=None
        repeated=relation_inputs(data,4,query,nodes,[.5],1.,.1,obs)
        self.assertTrue(torch.equal(result['pair'],repeated['pair']))
        obs[(2,0)]['frame']=4
        with self.assertRaises(ValueError):relation_inputs(data,4,query,nodes,[.5],1.,.1,obs)

    def test_real_pair_spaces_masks_null_and_gradients(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,x,soft=False):return torch.cat((self.weight*x['native_scores'],torch.zeros(1))),{}
        model=Model();one=lambda values:dict(native_scores=torch.tensor(values),current=torch.ones(1))
        features=dict(ab=[dict(B_index=0,A_indices=[0],features=one([.2])),dict(B_index=1,A_indices=[0,1],features=one([.4,.7]))],
            aa=one([.8,.3]),ba=one([.6,.1]),A_count=2,B_count=2,previous_time=2,current_time=3)
        masks=dict(legal_AB=torch.tensor([[True,False],[False,False]]),known_B=torch.tensor([True,False]),visible_A=torch.tensor([True,False]))
        loss,matrices=forward_cycle(model,features,masks)
        self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(torch.isfinite(model.weight.grad));self.assertGreater(abs(float(model.weight.grad)),0.)
        self.assertEqual(float(matrices['P_AB'][1,0]),0.);self.assertEqual(matrices['P_BA'][1].tolist(),[0.,1.])
        masks['visible_A']=torch.zeros(2,dtype=torch.bool);loss,_=forward_cycle(model,features,masks)
        self.assertEqual(float(loss),0.)


if __name__=='__main__':unittest.main()
