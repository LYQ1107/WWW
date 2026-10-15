"""Birth/Bank semantic labels, distinct from real native replay contracts."""
from gmt_jev_verifier_native_closure import classify
def main():
    sample=dict(id_count_before=20,refs=[10,20],past_history={10:1,20:2,7:1},
                GT_labels_OFFLINE_ONLY=[1],global_targets=[10])
    assert classify(sample,[21],[10])['category']=='APPLY_MLP'
    assert classify(sample,[10],[21])['category']=='KEEP_GMT'
    assert classify(sample,[21],[22])['category']=='EQUIVALENT_NATIVE_PLAN'
    assert classify(sample,[20],[21])['category']=='BOTH_WRONG'
    unknown=dict(sample,GT_labels_OFFLINE_ONLY=[None],global_targets=[None])
    assert classify(unknown,[21],[10])['label'] is None
    bank=dict(sample,global_targets=[7])
    assert classify(bank,[7],[21])['category']=='KEEP_GMT'
    assert classify(sample,[10],[5])['category']=='UNKNOWN_NATIVE_PLAN'
    print('NATIVE_LABEL_CONTRACTS_PASS: actual birth fragments, Bank target, unknown owner, semantic new-ID equivalence')
if __name__=='__main__':main()
