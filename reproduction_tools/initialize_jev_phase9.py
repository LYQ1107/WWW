"""Freeze supplemental sampling using ONLY existing observational indexes."""
import gzip,json,random,collections
from jev_phase9_common import *
def event_id(d):return tuple(d['key'])+(int(d['row']),)
def conflict_groups(records):
 # Transitive conflict graph; full candidate sets, same video, <=32 frames.
 # Same payload always connects. Shared target OR candidate ref connects.
 rows=sorted(records,key=event_id);parents=list(range(len(rows)))
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 def union(i,j):
  a,b=find(i),find(j)
  if a!=b:parents[max(a,b)]=min(a,b)
 for i,a in enumerate(rows):
  refs=set(a.get('candidate_track_ids',[]))|{a.get('proposal_track_id')}
  for j in range(i-1,-1,-1):
   b=rows[j]
   if a['key'][0]!=b['key'][0]:break
   if a['key'][1]-b['key'][1]>32:break
   bref=set(b.get('candidate_track_ids',[]))|{b.get('proposal_track_id')}
   same_target=a.get('offline_GT')is not None and a.get('offline_GT')==b.get('offline_GT')
   if a['key']==b['key']or same_target or(refs-{None})&(bref-{None}):union(i,j)
 result={};clusters={}
 for i,d in enumerate(rows):
  r=find(i);gid='V%d_F%d_VW%d_R%d'%event_id(rows[r]);result[event_id(d)]=gid;clusters.setdefault(gid,[]).append(d)
 return result,clusters

def main():
 protect();rng=random.Random(20261009);selected=[];sources=[];hard_all=[];oldkeys=set();byvideo={};naturalstats={}
 for v in TRAIN+VAL:
  scan=PREVIOUS/'opportunity_scan_v1'/f'video{v:02d}';m=json.loads((scan/'SCAN_RESULT.json').read_text())
  for s in m['bounded_snapshots']:
   oldkeys.update(tuple(s['key'])+(r,)for r in s['selected_rows'])
  hard=[json.loads(l)for l in gzip.open(scan/'hard_event_index.jsonl.gz','rt')];hard_all+=hard;byvideo[v]=hard
  for name in ('SCAN_RESULT.json','natural_event_index.jsonl.gz','hard_event_index.jsonl.gz'):sources.append({'path':str(scan/name),'sha256':sha(scan/name)})
  natural=[json.loads(l)for l in gzip.open(scan/'natural_event_index.jsonl.gz','rt')];naturalstats[v]=len(natural)
  strata={'NATURAL':natural,'HARD_NEGATIVE':[d for d in natural if d['proposal_correct']is True and d['first_action']=='ACCEPT_CURRENT'and d['top1_top2_margin']is not None and d['top1_top2_margin']<=0.1 and d['candidate_count']>=2],'UNKNOWN':[d for d in natural if d['offline_GT']is None or d['proposal_correct']is None]}
  for tag,pool in strata.items():
   cap={'NATURAL':8,'HARD_NEGATIVE':4,'UNKNOWN':2}[tag];draw=rng.sample(pool,min(cap,len(pool)))
   for i,d in enumerate(sorted(draw,key=event_id)):
    selected.append({'key':d['key'],'row':d['row'],'distribution':tag,'partition':'train'if v in TRAIN else'validation','sampling_probability':len(draw)/len(pool)if pool else None,'stratum_pool_size':len(pool),'counterfactual_audit':tag!='UNKNOWN'and i<({'NATURAL':2,'HARD_NEGATIVE':1}[tag]if tag!='UNKNOWN'else 0)})
 groupmap,groups=conflict_groups(hard_all)
 queues={};quotas=collections.Counter()
 for v in VAL:
  eligible=[d for d in byvideo[v]if event_id(d)not in oldkeys]
  vg=collections.defaultdict(list)
  for d in eligible:vg[groupmap[event_id(d)]].append(d)
  q=[]
  for g,pool in sorted(vg.items()):rng.shuffle(pool)
  while any(vg.values()):
   for g in sorted(vg):
    if vg[g]:q.append(vg[g].pop())
  queues[v]=q
 for i in range(48):
  active=[v for v in VAL if queues[v]]
  if not active:break
  v=min(active,key=lambda v:(quotas[v],v));d=queues[v].pop(0);quotas[v]+=1
  selected.append({'key':d['key'],'row':d['row'],'distribution':'SUPPLEMENTAL_CORRECTIVE','partition':'validation','group':groupmap[event_id(d)],'counterfactual_audit':True})
 # Quotas are fixed from group sizes, never future outcomes. Conditional SRS
 # within each group's selected quota gives known inclusion probability.
 n_by_g=collections.Counter(d['group']for d in selected if d['distribution']=='SUPPLEMENTAL_CORRECTIVE')
 eligible_by_g=collections.Counter(groupmap[event_id(d)]for v in VAL for d in byvideo[v]if event_id(d)not in oldkeys)
 for d in selected:
  if d['distribution']=='SUPPLEMENTAL_CORRECTIVE':d.update(sampling_probability=n_by_g[d['group']]/eligible_by_g[d['group']],stratum_pool_size=eligible_by_g[d['group']])
 protocol={'status':'FROZEN_BEFORE_NEW_FUTURE_OUTCOMES','base_phase8':BASE,'seed':20261009,'train':TRAIN,'validation':VAL,'heldout':[20,21,22],'heldout_sealed':True,'original13_unchanged':True,'observational_sources':sources,'group_rule':'same video; same payload OR within32frames with same anchored target OR shared full candidate track reference; transitive components; all current corrective availability events included','groups':{g:[list(event_id(x))for x in rows]for g,rows in groups.items()},'original_event_keys':[list(k)for k in sorted(oldkeys)],'selected_events':selected,'supplemental_budget':48,'actual_selected_supplemental':sum(quotas.values()),'supplemental_per_video':dict(quotas),'natural_population_per_video':naturalstats,'sampling':'seeded group round-robin SRS per fixed supplemental group quota, balanced video quota; strata SRS without replacement; all selected outcomes retained','group_weights':'one total weight per connected group; report raw count, independent units and Kish weight ESS separately; repeated events never independent','hard_negative_definition':'known anchored-correct factual ACCEPT, >=2 candidates, raw top1-top2 margin<=0.1; revalidate at captured native state','gate':{'train_verified':50,'supplemental_verified_ALONE':20,'combined_validation_verified':20,'train_videos':2,'val_videos':2,'train_groups':5,'val_groups':3,'natural_and_hard_negative_each_partition':True,'same_initial_complete_state_RNG_and_off_CONTROL':True,'actual_MATCH_candidate_commit_and_immediate_correct':True,'positive_complete_H32_and_birthzero':True,'no_GT_future_runtime':True},'unknown_candidate_utility':None,'horizons':[8,16,32],'utility':'Phase8 frozen correct-wrong-5merge-.25birth; all frozen sensitivities; UNKNOWN endpoint right-censored','future_policy':'live GMT OFF after each mutated commit','candidate_topk':'FULL, no GT-retention; TRAIN recall@8/16/32/64/full diagnostics','model_protocol':{'seeds':[20261008,20261009,20261010],'models':['CandidateMLP','DeepSets','CandidateJEV'],'parameter_target':34000,'epochs':[5,20,50,100],'data_group_fractions':[0.25,0.5,0.75,1],'normalization':['raw','train_stat','LayerNorm'],'learning_rate':0.001,'weight_decay':0.0001,'batch_size':32,'gradient_clip':5,'selection':'fixed budget validation protocol, strongest ordinary baseline; heldout released only after A/B/C/D PASS'},'Full24':False,'official_TEST':False,'phase8_fail_permanent':True}
 p=REPORTS/'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json';assert not p.exists();save(p,protocol)
 (ROOT/'docs/PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.md').write_text('# Frozen Phase IX supplemental capture protocol\n\nMachine protocol: reports/JEV_PHASE9/PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json. Frozen before any new future utility. Same train12/13/14/16 and validation17/18/19; heldout20/21/22 sealed. The Phase VIII13/13 result and FAIL remain unchanged.\n\nSupplemental ceiling48, balanced across videos and transitive conflict groups, with seeded sampling within fixed group quotas. Grouping uses full candidate-reference overlap, same-state rows and H32 temporal target dependence. Adjacent events may contribute labels but never count as independent replications; group weight totals one and Kish ESS is distinguished from independent groups. Require >=20 newly verified supplemental corrections independently of the old13. All failed/unknown/tie/wrong submissions retained.\n\nPer video also capture8 uniform natural rows,4 known-correct close-margin hard negatives (or all if fewer exist),2 unknown rows. Candidate correctness labels are separate from actual executed-branch future utility; unexecuted utility stays null. At most2 natural and1 negative row per video get bounded native branches, plus all supplemental rows. Full legal candidate sets; no GT selection in runtime. Counterfactual choices use GT only as offline audit interventions.\n\nOriginal train support is reused with original artifact hashes. Full initial mutable-state/RNG/control parity, legal actual candidate commit, immediate correction, complete positive H32 and birth-penalty-zero benefit must all hold. Supplemental-alone>=20, train>=50, >=2 videos per side, >=5/3 independent groups and complete natural/negative strata gate formal training. Full production candidate-values parity is a separate gate.\n\n**WHAT DID WE LEARN?** Existing observational indexes permit a sealed bounded supplement without changing splits or inspecting future causal outcomes. Effective support will be measured after native execution, never assumed from the48 budget.\n')
 print(json.dumps({'selected':len(selected),'supplemental':dict(quotas),'groups':len(groups),'protocol_sha256':sha(p)}))
if __name__=='__main__':main()
