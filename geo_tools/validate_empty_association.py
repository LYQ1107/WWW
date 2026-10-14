"""Actual checkpoint/GPU empty association and original RPCE RNG comparison."""
import numpy as np,torch,random,argparse
from common import *
from detectron2.structures import Instances,Boxes

def snapshot():
    return random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()

def restore(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3])

def equal_state(a,b):
    return a[0]==b[0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='empty_association_gpu_v2');args=parser.parse_args()
    cfg,model=load_gmt();rows=[]
    def inst(count):
        result=Instances((100,200));result.pred_boxes=Boxes(torch.tensor([[10.,20.,30.,40.]],device='cuda').repeat(count,1))
        # The final detector feature includes the spatial/temporal extension;
        # roi_heads.feature_dim alone is only the appearance sub-dimension.
        result.reid_features=torch.randn(count,model.roi_heads.transformer.d_model,device='cuda')
        return result
    with torch.inference_mode():
        for history_count,query_count in [(2,2),(2,0),(0,2),(0,0)]:
            history,query=inst(history_count),inst(query_count);history.track_ids=torch.arange(1,history_count+1,device='cuda')
            instances=[history,query];features=torch.cat([history.reid_features,query.reid_features])[None]
            before=snapshot();legacy=None;error=None
            try:
                legacy=model.roi_heads._forward_transformer(instances,features,None,1,None,None,history.track_ids.float())
                torch.cuda.synchronize()
            except (RuntimeError,ValueError) as exc:
                error=f'{type(exc).__name__}: {exc}'
                if history_count and query_count:raise
                if 'must match the size' in str(exc):raise AssertionError('invalid feature-dimension fixture') from exc
            legacy_state=snapshot();restore(before)
            output,boxes,counts,npast,indices=model.get_asso(instances,1)
            torch.cuda.synchronize();new_state=snapshot()
            same_rng=equal_state(legacy_state,new_state)
            expected_shape=(query_count,history_count)
            same_values=legacy is None or (torch.equal(legacy[0][0],output[0]) and torch.equal(legacy[1],boxes))
            if output[0].shape!=expected_shape or not same_rng or not same_values:
                raise AssertionError('empty association or legacy RNG contract changed')
            rows.append(dict(history_count=history_count,query_count=query_count,output_shape=list(output[0].shape),
                legacy_transformer_error=error,legacy_rng_parity=same_rng,legacy_values_parity=(same_values if legacy is not None else None),new_path_success=True))
    report=dict(status='PASS',scope='real fixed GMT checkpoint on actual GPU; original ROI transformer empty calls compared against get_asso bypass; all four RNG streams',cases=rows,provenance=provenance())
    save_json(ROOT/f'results/{args.tag}.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
