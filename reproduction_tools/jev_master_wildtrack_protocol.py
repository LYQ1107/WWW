"""Freeze the exact previously authorized Phase XIV TRAIN prefix, no new data."""
from jev_master_audit_common import ROOT, MASTER, protect, read, ref, save, sha
from pathlib import Path


def main():
    protect()
    data = Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1/external_wildtrack_v1')
    previous = Path('/home/liuyeqiang/WWW_jev_phase14')
    old = previous/'reports/JEV_PHASE14/EXTERNAL_GENERALIZATION_PROTOCOL.json'
    scope = previous/'reports/JEV_PHASE14/EXTERNAL_DATA_SCOPE_AUDIT.json'
    result = read(data/'RESULT.json')
    directory = read(data/'ARCHIVE_DIRECTORY.json')
    # Exclude macOS resource forks: they are not annotation members.
    names = sorted(n for n in directory['names']
                   if n.startswith('Wildtrack_dataset/annotations_positions/')
                   and n.endswith('.json'))
    assert len(names) == 400
    records = sorted((x for x in result['manifests']
                      if x['archive_member'].startswith('Wildtrack_dataset/annotations_positions/')),
                     key=lambda x:x['archive_member'])
    assert len(records) == 320
    assert [x['archive_member'] for x in records] == names[:320]
    for x in records:
        assert sha(x['path']) == x['SHA256']
        assert Path(x['path']).stat().st_size == x['bytes']
    size = sum(x['bytes'] for x in records)
    assert size < 64*2**20
    save(MASTER/'WILDTRACK_REUSED_TRAIN_PROTOCOL.json', dict(
        status='FROZEN_BEFORE_NEW_SEMANTIC_GT_AUDIT',
        discovery_disclosure='Phase14 published counts and schema examples already known; reused TRAIN audit, not unseen evaluation/performance preregistration. Byte hashes read for freeze.',
        scope_authority='Reuse earlier Phase14 authorized first320 labels; published necessity/split/resource proposal and MASTER13/17. No new dataset labels/pixels downloaded; no extra approval for routine read-only reuse.',
        previous_proposal=ref(Path('/home/liuyeqiang/WWW_jev_phase21/docs/PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md')),
        old_protocol=ref(old), old_scope_correction=ref(scope),
        old_download_result=ref(data/'RESULT.json'),
        archive_directory=ref(data/'ARCHIVE_DIRECTORY.json'),
        GT_records=[{k:x[k] for k in ['archive_member','path','SHA256','bytes']} for x in records],
        source_path_search=ref(Path('/home/liuyeqiang/WWW_jev_master_audit_runtime/LOCAL_SOURCE_PATH_SEARCH.json')),
        published_split_evidence=dict(
            url='https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1562-supp.pdf',
            section='5 Recommended splits, supervised scenario', TRAIN=360, TEST=40,
            local_PDF=ref(Path('/home/liuyeqiang/WWW_jev_master_audit_runtime/WILDTRACK_OFFICIAL_SUPPLEMENT.pdf')),
            verification='Primary author-hosted search excerpt and old frozen protocol. Local PDF retained; pdftotext unavailable, no package installed.'),
        unread_TRAIN_tail_names=names[320:360], sealed_TEST_names=names[360:],
        roles=dict(existing_whole_prefix='TRAIN_AUDIT_ONLY_PREVIOUSLY_EXPOSED',
                   new_RESERVED=[], same_capture_camera_or_temporal_split_allowed=False),
        audit_spec=['Raw duplicate person/view IDs, no cleanup',
                    'Exact absent sentinel versus mixed -1 and bbox geometry/intersection',
                    'Persistent declared IDs/gaps/cross-view projection schema, no visual ID certification',
                    'Co-occurrence diagnostic, not causal safe-group independence',
                    'Shared7-view GT scope disclosed, selected C1/C2 interface separately counted'],
        budget=dict(CPU_threads=1,GPU_seconds=0,existing_GT_bytes_to_audit=size,
                    max_GT_bytes=64*2**20,new_output_bytes_max=2*2**20,
                    training=False,native_interventions=0,new_images_read=0,new_dataset_downloads=0),
        protected=['all V-XVI and XVII-XXI originals','old Phase14 labels immutable',
                   'TRAIN tail40 not selected','TEST40 sealed','VisionTrack20/21/22','Full24'],
        qualification='Structural GT checks do not certify visual identity, safe actions, independent RESERVED or learning.'))
    (ROOT/'docs/WWW_JEV_REUSED_WILDTRACK_SCOPE.md').write_text('''# 既有 WILDTRACK TRAIN 输入审计范围

本轮复用 Phase XIV 已授权并留存的前320标注时刻。必要性：独立监督不足，需要先核查既有外部身份标签的结构与可用性。MASTER第13节要求先说明必要性、划分、资源，第17节允许普通工程主动执行；历史待答问卷不阻挡这项既有数据的只读审计。

此前公开来源审计没有定位到这份既有副本，本轮目录搜索找到它。它已用于Phase XIV离线评价，必须标记历史暴露。整个前缀固定为TRAIN审计人口，不能按相机、时间尾段或人员计数改称新RESERVED；不挑有利样本、不训练、不运行新轨迹。

作者补充材料的监督方案使用前360标注时刻训练，最后40测试。既有副本只有前320份共享GT，文件00000000至00001595；后40 TRAIN和最后40 TEST正文不读取。论文角色优先于加载器中的val变量名。

已有GT约10MiB，原地只读、逐文件哈希冻结。不读图像、不复制标签、不新下载数据、不安装环境，0GPU，新增输出不超过2MiB。共享JSON包含七相机投影，审计如实披露；未来原生兼容接口仍只有C1/C2。结构统计不证明人工身份绝对正确、因果收益或8＋3学习资格。详细固定文件和未读边界见WILDTRACK_REUSED_TRAIN_PROTOCOL.json；所有旧结果不变。
''')
    print('FROZEN_REUSED_TRAIN',len(records),size)


if __name__ == '__main__':
    main()
