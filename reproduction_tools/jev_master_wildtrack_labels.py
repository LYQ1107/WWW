"""Read only the frozen reused TRAIN labels. No image/actor/GT cleanup."""
from collections import Counter, defaultdict
import math
import subprocess
from pathlib import Path
from jev_master_audit_common import ROOT, MASTER, protect, read, save, ref, sha


def main():
    protect()
    p = MASTER/'WILDTRACK_REUSED_TRAIN_PROTOCOL.json'
    protocol = read(p)
    for key in ['old_protocol','old_scope_correction','old_download_result','archive_directory','source_path_search']:
        item=protocol[key];assert sha(item['path'])==item['SHA256']
    records=protocol['GT_records'];assert len(records)==320
    assert sum(x['bytes'] for x in records)<=protocol['budget']['max_GT_bytes']
    assert not ({x['archive_member'] for x in records} & set(protocol['sealed_TEST_names']+protocol['unread_TRAIN_tail_names']))
    counts=Counter();selected=Counter();perview=defaultdict(Counter)
    identities=defaultdict(list);frames=[];problems=[];parents={}

    def find(a):
        parents.setdefault(a,a)
        while parents[a]!=a:
            parents[a]=parents[parents[a]];a=parents[a]
        return a

    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[b]=a

    stamps=[]
    for frame,item in enumerate(records):
        path=Path(item['path'])
        assert sha(path)==item['SHA256'] and path.stat().st_size==item['bytes']
        stamps.append(int(path.stem));raw=read(path);assert isinstance(raw,list)
        ids=set();frame_bad=False;counts['person_records']+=len(raw)
        for row in raw:
            ident=row.get('personID');position=row.get('positionID');views=row.get('views')
            if type(ident) is not int or ident<0 or type(position) is not int or not 0<=position<480*1440 or not isinstance(views,list):
                counts['bad_record_schema']+=1;frame_bad=True;continue
            if ident in ids:counts['duplicate_person_per_frame']+=1;frame_bad=True
            ids.add(ident);identities[ident].append(frame);find(ident)
            seen=set()
            for box in views:
                view=box.get('viewNum');coords=[box.get(k) for k in ['xmin','ymin','xmax','ymax']]
                if type(view) is not int or view not in range(7) or view in seen:
                    counts['bad_or_duplicate_view']+=1;frame_bad=True;continue
                seen.add(view);c=Counter();c['projections']+=1
                if not all(type(x) in [int,float] and math.isfinite(x) for x in coords):
                    c['nonfinite_or_missing_bbox']+=1;frame_bad=True
                elif coords==[-1,-1,-1,-1]:c['exact_absent_sentinel']+=1
                else:
                    x,y,x2,y2=coords
                    if -1 in coords:c['mixed_minus1_coordinate']+=1
                    if x2<=x or y2<=y:c['nonpositive_area']+=1;frame_bad=True
                    elif min(x2,1920)<=max(x,0) or min(y2,1080)<=max(y,0):c['outside_image']+=1
                    else:
                        c['positive_area_with_image_intersection']+=1
                        if -1 in coords:c['mixed_minus1_with_valid_intersection']+=1
                        if x<0 or y<0 or x2>1920 or y2>1080:c['partially_outside']+=1
                perview[view].update(c);counts.update(c)
                if view in [0,1]:selected.update(c)
            if seen!=set(range(7)):counts['incomplete_view_set']+=1;frame_bad=True
        # Deliberately conservative simultaneous-person graph: not a native causal graph.
        if ids:
            first=min(ids)
            for ident in ids:union(first,ident)
        frames.append(dict(frame=frame,people=len(ids),structural_schema_issue=frame_bad))
        if frame_bad:problems.append(path.name)
    assert stamps==list(range(0,1600,5))
    components=Counter(find(i) for i in parents)
    repeated=sum(len(v)>1 for v in identities.values())
    gaps=[b-a-1 for vv in identities.values() for a,b in zip(sorted(set(vv)),sorted(set(vv))[1:])]
    result=dict(status='COMPLETE_REUSED_TRAIN_LABEL_AUDIT',
        actor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        actor=ref(Path(__file__)),protocol=ref(p),
        input=dict(files=320,bytes=sum(x['bytes'] for x in records),first_stamp=stamps[0],last_stamp=stamps[-1],
                   measured_stamp_step=5,published_annotation_Hz=2,GT_interpolation=False,
                   scope='Previously exposed shared7-view GT; selected-interface statistics C1/C2'),
        all_views=dict(counts),selected_C1_C2=dict(selected),per_view={str(k):dict(v) for k,v in perview.items()},
        raw_structural_issue_frames=problems,raw_labels_changed=False,
        identity=dict(declared_distinct=len(identities),declared_repeated=repeated,
                      frame_person_min=min(x['people'] for x in frames),frame_person_max=max(x['people'] for x in frames),
                      positive_gap_transitions=sum(x>0 for x in gaps),max_gap_observed_frames=max(gaps,default=0),
                      manual_visual_identity_correctness='NOT_VERIFIED',cross_view_consistency='Declared shared-ID schema only, no visual certification'),
        dependency_diagnostic=dict(method='All co-present GT IDs joined transitively; not actual Hungarian/native causal dependency',
                                   components=len(components),component_sizes=sorted(components.values(),reverse=True),
                                   independent_captures=1,new_reserved_captures=0,
                                   safe_causal_action_groups='NOT_MEASURED'),
        legacy_input_risk='Phase14 prepare skips any bbox coordinate == -1. Exact all-four absence versus mixed -1 counts reported without repairing old GT/results.',
        source_role='REUSED_TRAIN_ONLY_NOT_FRESH_VALIDATION',
        learning_qualified=False,old_8_plus_3_gate_unchanged=True,
        new_images_read=0,new_dataset_downloads=0,GPU_seconds=0,training=False,native_interventions=0,
        unread=['TRAIN tail40 annotation bodies','TEST40 annotation bodies','images','old tracking scores','VisionTrack20/21/22'],
        next_action='Need a separate lawful manual validation cohort; do not invent independence from views/frames/declared ID counts.')
    save(MASTER/'WILDTRACK_REUSED_TRAIN_LABEL_AUDIT.json',result)
    source=dict(status='EXISTING_REUSED_TRAIN_COPY_LOCATED',path_search=protocol['source_path_search'],
                old_protocol=protocol['old_protocol'],old_scope_correction=protocol['old_scope_correction'],
                label_manifest=ref(p),download_needed=False,old_external_exposure=True,
                public_access_basis='Earlier Phase14 official EPFL release acquisition; reuse same320 files. No new data rights asserted.',
                MMPTRACK='No lawful local copy located within bounded directory search; no global absence claim',
                limitation='This available TRAIN copy does not supply independent RESERVED supervision.')
    save(MASTER/'EXISTING_SOURCE_ACCESS_AUDIT.json',source)
    (ROOT/'docs/WWW_JEV_REUSED_WILDTRACK_LABEL_AUDIT.md').write_text(f'''# 既有 WILDTRACK TRAIN 标注审计

发现并复用了Phase XIV已留存的前320份原始TRAIN共享标注，正文共{result['input']['bytes']}字节。每份SHA256与旧下载manifest一致；文件名从00000000到00001595，步长5，对应发布的2Hz标注时刻。未读图像、后40 TRAIN或最后40 TEST标注；没有下载数据、训练或GPU任务。

共有{len(identities)}个声明personID，其中{repeated}个跨帧重复；这些是标签ID数量，不能当作独立因果安全组。原始person/frame重复键{counts['duplicate_person_per_frame']}，视角键异常{counts['bad_or_duplicate_view']}，不完整七视角集合{counts['incomplete_view_set']}，存在结构问题的原始帧{len(problems)}。不清除重复或改写GT；机械结构通过也不证明每条人工身份绝对正确。

共享GT实际含七相机投影，C1/C2单独统计。C1/C2有{selected['mixed_minus1_with_valid_intersection']}个混合-1且正面积、有图像交集的框。旧准备脚本遇到任一坐标为-1就跳过；这里记录其与完整四坐标缺席哨兵的区别，不修订旧评价结果。全部缺席/边界/无交集计数见JSON。

同帧人物共现的保守连通诊断得到{len(components)}个分量，最大{max(components.values(),default=0)}个ID。它不是原生Hungarian因果依赖图，也不能认证独立训练组。数据只有一个连续采集，整个前缀已作过离线评价，本轮固定为历史暴露的TRAIN审计人口；新RESERVED采集数为0。不按相机或时间再划出“独立验证”。

本轮确认了可复用的TRAIN路径与结构，仍未建立安全动作监督或独立RESERVED。原3 TRAIN/1 RESERVED、8＋3资格不变，不能启动依赖该资格的训练。下一步需要独立、合法的人工验证来源；MMPTRACK若有合法副本可继续审计。目录搜索有深度和数量边界，未找到副本不代表整机绝对不存在。

作者监督分割以最后40标注时刻为TEST，加载器的val变量不能解除保护：[补充材料第5节](https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1562-supp.pdf)。协议、原始计数、未读边界与哈希均已保存；未使用旧跟踪分数挑选输入。
''')
    print('COMPLETE',len(identities),repeated,'structural_issue_frames',len(problems),
          'mixed_minus1_C1C2',selected['mixed_minus1_with_valid_intersection'])


if __name__=='__main__':
    main()
