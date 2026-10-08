"""Offline prefix-anchored effects; deliberately absent from live actors."""
from collections import Counter,defaultdict

def effects(rows,prefix,start_frame,horizon,target_gts):
    mapping={int(t):int(g)for t,g in prefix['reliable'].items()}
    prefix_counts={int(t):Counter({int(g):n for g,n in c.items()})for t,c in prefix['counts'].items()}
    observed=defaultdict(Counter);born={};merges=set();births=set();c=Counter();target=Counter();timelines=defaultdict(dict)
    known_gts=set(int(g)for g in prefix['last_GT_frame'])
    for key,r in sorted(rows.items()):
        frame,view,row=key
        if not start_frame<=frame<start_frame+horizon:continue
        g=r['gt'];t=int(r['id'])
        if g is None:c['GT_unknown']+=1;continue
        g=int(g);observed[t][g]+=1
        if t not in prefix_counts:
            born.setdefault(t,g)
            if g in known_gts:births.add(t)
            if sum(observed[t].values())==2 and len(observed[t])==1:mapping[t]=born[t]
        identity=mapping.get(t)
        kind='anchor_unknown'if identity is None else'correct'if identity==g else'wrong'
        c[kind]+=1
        if g in target_gts:target[kind]+=1
        if identity is not None and identity!=g:merges.add(t)
        if t not in prefix_counts and len(observed[t])>1:merges.add(t)
        # An unknown identity is an unobserved error status, not recovery.
        # If duplicated annotation identities produce multiple rows, any
        # proven wrong observation dominates; otherwise uncertainty remains.
        status=None if kind=='anchor_unknown'else kind=='wrong'
        previous=timelines[(g,view)].get(frame,False)
        timelines[(g,view)][frame]=True if previous is True or status is True else None if previous is None or status is None else False
    c['false_merge_tracks']=len(merges);c['false_births']=len(births)
    episodes=[]
    for (g,v),timeline in sorted(timelines.items()):
        run=[];previous=None
        for f,bad in sorted(timeline.items()):
            if run and(bad is not True or f!=previous+1):
                episodes.append({'gt':g,'view':v,'start':run[0],'end':run[-1],'duration_frames':len(run),'right_censored':bad is None or f!=previous+1});run=[]
            if bad is True:run.append(f)
            previous=f
        if run:episodes.append({'gt':g,'view':v,'start':run[0],'end':run[-1],'duration_frames':len(run),'right_censored':True})
    def utility(wb,wm,ww):return c['correct']-ww*c['wrong']-wm*c['false_merge_tracks']-wb*c['false_births']
    return {'counts':dict(c),'target_counts':dict(target),'utility':utility(.25,5,1),'utility_birth_zero':utility(0,5,1),
        'sensitivity':{f'b{wb}_m{wm}_w{ww}':utility(wb,wm,ww)for wb in(0,.25,1)for wm in(2,5)for ww in(1,2)},
        'wrong_identity_episodes':episodes,'wrong_identity_duration_camera_frames':sum(e['duration_frames']for e in episodes),
        'definition':'Permanent pre-intervention confirmed anchors; previously unanchored existing IDs stay UNKNOWN; newly born IDs confirm only after first two consistent known observations; two-camera frame episodes, no FPS assumed'}
