"""Small but substantive UNKNOWN, dependency and whole-plan safety tests."""
from gmt_jev_verifier_labels import *
def main():
    assert history_label(2,{1:2},None,1)=='UNKNOWN'
    assert history_label(6,{1:3,2:3},1,1)=='CONFLICTED'
    assert history_label(4,{1:4},1,2)=='UNKNOWN'
    assert history_label(4,{1:4},1,1)=='RELIABLE'
    def plan(a,b,targets=[0,1]):return paired_plan(a,b,[10,20],targets,['CERTIFIED_UNIQUE_ACTIVE']*2,[1,2],{10:1,20:2})
    assert plan([0,1],[1,0])['label']==0
    assert plan([1,0],[0,1])['label']==1
    assert plan([0,1],[-1,1])['category']=='UNKNOWN_PLAN'
    assert plan([0,1],[1,0],[-1,1])['label'] is None
    assert plan([0,1],[0,1])['label'] is None
    one=paired_plan([0],[1],[10,20],[0],['CERTIFIED_UNIQUE_ACTIVE'],[1],{10:1,20:2})
    assert one['people']==[1,2], 'Include both candidate owners in dependence groups'
    try:plan([0,1],[0,0]);raise AssertionError('Invalid occupied ID plan was accepted')
    except AssertionError as e:assert 'splice' in str(e)
    events=[dict(video=12,people=[1,2]),dict(video=12,people=[2,3]),dict(video=12,people=[4])]
    assert dependency_components(events)==2
    print('LABEL_CONTRACTS_PASS: short/alias UNKNOWN, real conflict, Bank UNKNOWN, complete legal plans, person dependence')
if __name__=='__main__':main()
