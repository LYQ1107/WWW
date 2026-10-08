"""Synthetic engineering tests: no frozen research data or heldout read."""
import gzip,json,tempfile
from pathlib import Path
import torch
from jev_phase10_common import REPORTS,save
from jev_phase10_online_audit import audit
from jev_phase10_learning import evaluate,normalization,install_normalization,build_candidate_model


def main():
 aligned={};commits=[];packets=[]
 for frame,ids,gts in [(0,[1,2],[10,20]),(1,[1,2],[10,20]),(2,[1],[10]),(3,[1],[20]),(4,[3],[10]),(5,[3],[10]),(6,[3],[10]),(7,[1],[None])]:
  for row,(t,g)in enumerate(zip(ids,gts)):aligned[frame,0,row]={'id':t,'gt':g}
  commits.append({'key':[12,frame,0],'ids':ids,'events':[{'birth':frame in [0,4],'write':True}for t in ids]})
  if frame in [2,3]:packets.append({'key':[12,frame,0],'candidate_ids':[1,2],'GMT_scores':[[1.,3.]],'thresholds':[0.,0.],'legal_mask':[[True,True]],'candidate_values':[[3.,1.]],'initial_proposal_ids':[2],'existing_ids':[1]})
 with tempfile.TemporaryDirectory(prefix='phase10_audit_contract_')as tmp:
  paths=[]
  for name,values in [('journal',packets),('commits',commits)]:
   path=Path(tmp)/(name+'.jsonl.gz')
   with gzip.open(path,'wt')as f:
    for value in values:f.write(json.dumps(value)+'\n')
   paths.append(path)
  result=audit(aligned,*paths);c=result['counts'];assert c['N01_same_prefix_fixed_to_policy']==1 and c['N10_same_prefix_fixed_to_policy']==1;assert c['wrong_gallery_writes']==1 and c['false_birth_observations']==1;assert result['wrong_ID_duration_total_camera_frames']==1;assert result['candidate_rank1']==.5 and result['candidate_MRR']==.75;assert result['row_statuses']['4,0,0']is None and result['row_statuses']['5,0,0']is None and result['row_statuses']['6,0,0']is True and result['row_statuses']['7,0,0']is None
 from detectron2.structures import Instances,Boxes
 from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
 inst=Instances((100,200));inst.pred_boxes=Boxes(torch.tensor([[10.,20.,30.,40.]]));inst.track_ids=torch.tensor([7]);inst.scores=torch.ones(1);inst.pred_classes=torch.zeros(1,dtype=torch.long);processed=CustomRCNN._postprocess([inst],[{'height':200,'width':400}],[(0,0)],not_clamp_box=True)[0]['instances'];assert torch.equal(processed.pred_boxes.tensor,torch.tensor([[20.,40.,60.,80.]]))and processed.track_ids.tolist()==[7]
 rows=[]
 for i in range(4):rows.append({'key':[12,i,0],'row':0,'candidate_ids':(1,2,3),'state64':torch.ones(64)*i,'evidence12':torch.randn(3,12),'legal_mask':torch.ones(3,dtype=torch.bool),'known_mask':torch.tensor([True,True,False]),'positive_mask':torch.tensor([True,False,False]),'utility':torch.tensor([1.,0.,float('nan')]),'utility_mask':torch.tensor([True,True,False]),'weight':1.,'group':str(i),'temporal_dependency_bundle':str(i),'distribution':['NATURAL'],'factual_identity_correct':False,'partition':'train','CE_eligible':True,'ranking_eligible':True})
 for name in ['CandidateMLP','CandidateDeepSets','CandidateJEV']:
  model=build_candidate_model(name);install_normalization(model,normalization(rows));evaluation=evaluate(model,rows,'joint');assert evaluation['selection_NLL']is not None and evaluation['executed_pair_count']==4 and 0<=evaluation['ECE10']<=1
 save(REPORTS/'CLOSED_LOOP_AUDIT_CONTRACT_TESTS.json',{'status':'PASS','synthetic_permanent_prefix_anchors':True,'unknown_birth_not_counted_correct':True,'N01_N10_separate':True,'actual_wrong_camera_frame_duration':True,'wrong_write_false_birth':True,'actual_CustomRCNN_dimension_postprocess':True,'three_model_stable_NLL_ECE_executed_pair_evaluation':True,'research_training_performed':False,'heldout_read':False});print('CLOSED_LOOP_AUDIT_ENGINEERING_PASS')
if __name__=='__main__':main()
