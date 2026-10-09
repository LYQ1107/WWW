"""Byte-exact CPU/CUDA real-prefix token optimization, no metric-based tuning."""
import dataclasses,torch
from jev_phase12_common import *
from gtr.modeling.visual_jev_mcmot import identity_history_tokens as original,fast_identity_history_tokens as fast

def main():
    protect();source=binding();assert not source['dirty'];torch.set_num_threads(1);checks=[]
    for video in VAL:
        entries=sorted(json.loads((PREVIOUS/'native_capture_v1'/f'video{video:02d}/compat/RESULT.json').read_text())['prefixes'],key=lambda e:e['key'])
        for entry in [entries[0],entries[len(entries)//2],entries[-1]]:
            frame,view=entry['key'][1:]
            for device in ['cpu','cuda:0']:
                pkt=torch.load(entry['packet_path'],map_location=device)['packet']['batch'];prefix=torch.load(entry['path'],map_location=device);obs=prefix['instances'][frame*2+view].reid_features
                a=original.native_time_metadata(prefix['instances'],frame,view,first=prefix['first']);b=fast.native_time_metadata(prefix['instances'],frame,view,first=prefix['first']);assert a==b
                for terminal in [False,True]:
                    x=original.build_match_inputs(pkt,obs,prefix['galleries'],frame=frame,view=view,metadata=a,include_terminal=terminal);y=fast.build_match_inputs(pkt,obs,prefix['galleries'],frame=frame,view=view,metadata=b,include_terminal=terminal)
                    for one,two in zip(x,y):
                        for field in dataclasses.fields(one):assert torch.equal(getattr(one,field.name),getattr(two,field.name)),(entry['key'],device,field.name)
                checks.append({'key':entry['key'],'device':device,'all_state_question_option_tensors_bitwise_equal':True,'metadata_equal':True})
    save(OUT/'fast_tokens_contract_v1/RESULT.json',{'status':'PASS','binding':source,'checks':checks,'scope':'predeclared first/middle/last actual prefix per validation video; CPU/CUDA, with and without terminal; no mathematical/NN/action change; main online experiments remain original frozen implementation'})
    print('REAL_NATIVE_TOKEN_BATCHING_BITWISE_PASS',flush=True)
if __name__=='__main__':main()
