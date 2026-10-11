"""Inspect frozen public source snapshots only; never import dataset loaders."""
import ast
import json
import subprocess
from pathlib import Path
from jev_master_audit_common import ROOT, MASTER, protect, read, ref, save, sha


def main():
    protect()
    protocol_path = MASTER / 'PUBLIC_RELEASE_METADATA_PROTOCOL.json'
    protocol = read(protocol_path)
    snapshot = Path(protocol['local_snapshot'])
    files = protocol['snapshot_files']
    actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file()}
    assert actual == set(files), ('unexpected snapshot file', actual ^ set(files))
    for name, expected in files.items():
        assert sha(snapshot / name) == expected, name
    size = sum((snapshot / name).stat().st_size for name in files)
    assert size <= protocol['budget']['source_snapshot_MiB_max'] * 2**20
    checks = []

    def check(name, marker, meaning):
        text = (snapshot / name).read_text()
        assert marker in text, (name, marker)
        pos = text.index(marker)
        checks.append(dict(file=name, SHA256=files[name],
                           line=text[:pos].count('\n') + 1, meaning=meaning))

    umpn = 'cvlab-epfl/umpn/'
    toolkit = 'cvlab-epfl/scout_toolkit/'
    check(umpn+'configs/files/config_scout.yaml', '- scoutval',
          'Released example trains using the dataset named scoutval.')
    check(umpn+'configs/files/config_wildtrack.yaml', '- wildtrack',
          'Released WILDTRACK example uses this same dataset for temporal splitting.')
    for name in ['config_scout.yaml', 'config_wildtrack.yaml']:
        check(umpn+'configs/files/'+name, "--split_proportion: '0.9'",
              'Example requests a 90/10 split, not participant/session disjointness.')
    check(umpn+'datasets/factory.py', '"sequence_1", cam_name=[0, 1, 2, 3, 4, 5, 6, 7]',
          'scoutval selects sequence_1 cameras 0 through 7.')
    check(umpn+'datasets/factory.py', '"sequence_1", cam_name=[8, 9, 10',
          'scouttrain also includes the same sequence, using other cameras.')
    check(umpn+'datasets/utils.py', 'train_idx = list(range(0, train_size))',
          'TRAIN is a prefix of dataset indices.')
    check(umpn+'datasets/utils.py', 'test_idx = list(range(len(dataset_val) - val_size, len(dataset_val)))',
          'The variable test_idx supplies the internal validation tail; no official TEST data was opened.')
    check(umpn+'datasets/factory.py', 'get_train_val_split_index(dataset, dataset_val,',
          'The split function is actually used by the dataloader factory.')
    check(umpn+'datasets/sceneset.py', 'for index in range(self.__len__()):',
          'Missing normalization statistics scan the entire scene before any external restriction.')
    check(umpn+'datasets/sceneset.py', 'frame_dict = self.get(index, view_id)',
          'The statistics scan uses the scene getter; restrict fit to frozen TRAIN in any new adapter.')
    check(umpn+'datasets/sceneset.py', 'self._get_gt(index, view_id)',
          'GT-derived fields exist in the getter; do not feed them into an online policy.')
    check(umpn+'datasets/scout.py', 'geometry.project_2d_points_to_mesh',
          'Detector bottom-center projection via calibration is separate from annotation world points.')
    check(toolkit+'src/loading.py', "'track_id':ann['category_id']",
          'Official COCO reader interprets category_id as identity; generic person-category assumptions are unsafe.')
    check(toolkit+'src/loading.py', 'timestamps_sequence_{sequence}.json',
          'Use explicit per-camera timestamps; nominal viewer FPS is insufficient.')
    check('scout_frontend.js', 'complete annotations for the first 8 cameras of sequence 1',
          'Release announcement limits complete annotations to sequence_1 first eight cameras.')
    check('scout_frontend.js', 'pseudo-labels extracted automatically',
          'Other camera annotations must not be treated as equivalent manual identity supervision.')
    check('mmp_release.html', '14 in training, 7 in validation and 7 in testing',
          'Official page declares participant split counts; actual cohort uniqueness remains unverified.')
    check('mmp_release.html', 'download and sign',
          'Dataset access requires the recipient to sign terms and contact the provider.')
    # AST parsing checks syntax, without importing or executing third-party source.
    parsed = []
    for name in files:
        if name.endswith('.py'):
            ast.parse((snapshot / name).read_text(), filename=name)
            parsed.append(name)
    metadata = read(snapshot / 'scout_metadata.json')
    assert set(metadata) == {'dataset_name', 'sequences'}
    for seq in metadata['sequences']:
        assert set(seq) == {'id','description','duration_sec','fps','num_cameras','num_frames'}
        assert seq['fps'] == 10.0
    head = read(snapshot / 'wildtrack_archive_head.json')
    assert head['method'] == 'HEAD' and head['archive_body_read'] is False
    archive_size = int(head['headers']['Content-Length'])
    assert archive_size == 6807496358
    sources = []
    for repo in protocol['repositories']:
        for path in repo['paths']:
            name = repo['repo']+'/'+path
            sources.append(dict(url=f"https://raw.githubusercontent.com/{repo['repo']}/{repo['commit']}/{path}",
                                snapshot=name, SHA256=files[name]))
    for name, url in protocol['official_pages'].items():
        sources.append(dict(url=url, snapshot=name, SHA256=files[name]))
    sources.extend([
        dict(url=protocol['website_bundle']['url'], snapshot='scout_frontend.js', SHA256=files['scout_frontend.js']),
        dict(url=protocol['metadata_urls'][0], snapshot='scout_metadata.json', SHA256=files['scout_metadata.json'])])
    result = dict(
        status='COMPLETE_PUBLIC_SOURCE_REVIEW_ONLY',
        actor_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        actor=ref(Path(__file__)), protocol=ref(protocol_path),
        snapshot_bytes=size, snapshot_files=len(files), syntax_checked_files=parsed,
        sources=sources, assertions=checks,
        source_discovery_before_protocol=True,
        dataset_qualification='NOT_CERTIFIED_WITHOUT_LICENSED_PROSPECTIVE_TRAIN_VAL_GT_AUDIT',
        candidates={
            'MMPTRACK': dict(priority='PREFERRED_IF_LAWFUL_COPY_AVAILABLE',
                participant_counts=dict(TRAIN=14, VAL=7, TEST=7), fps=15,
                access='Recipient-signed terms; no legitimate local copy established, no message/signature sent',
                identity_scope='Per-environment IDs versus separately requested cross-environment retrieval IDs must be audited',
                helper_license='MIT applies to helper code only, not dataset terms'),
            'WILDTRACK': dict(gt_frames_reported=400, gt_fps=2, released_image_fps=10, raw_video_fps=60,
                complete_archive_bytes=archive_size, archive_size_method='HEAD_ONLY',
                acquisition='One published camera setup; seven cameras are not seven independent captures',
                split='UMPN example 90/10 temporal; no participant-disjoint enforcement shown',
                identity='personID distinct from positionID; persistence/uniqueness not checked',
                label_interpolation=False, data_specific_license='NOT_VERIFIED'),
            'SCOUT': dict(complete_annotation_release_scope='sequence_1 cameras 0..7 per website announcement',
                actual_manual_quality='NOT_VERIFIED_WITHOUT_LABELS_AND_PROVENANCE',
                other_annotations='AUTO_PSEUDO_NOT_ELIGIBLE_AS_MANUAL_SUPERVISION',
                example_train_source='scoutval sequence_1 cameras0..7 then 90/10 temporal split',
                independent_split='NOT_CERTIFIED; other camera partitions share sequence_1',
                converted_viewer_metadata=metadata,
                raw_archive_fps='NOT_VERIFIED; converted metadata is not raw archive timing',
                loader_ID_semantics='COCO category_id becomes track_id',
                missing_labels='Missing-file versus genuine empty frame must be distinguished in an adapter',
                data_specific_license='NOT_VERIFIED; JS dependency licenses do not grant dataset rights')},
        engineering_requirements=[
            'Freeze whole capture/cohort roles before label results; never count cameras or frames as independent groups.',
            'Do not infer actual participant overlap from a temporal split alone; audit identity cohorts on legal labels.',
            'Fit all normalization on TRAIN only; prohibit GT world positions/GT-filled detections as policy inputs.',
            'Verify real detector availability, score semantics, timestamp alignment and manual ID persistence.',
            'No automatic image/model download; select small labels separately only after lawful scope is established.'],
        learning_gate_unchanged=dict(TRAIN_independent_safe=3, RESERVED_independent_safe=1,
                                     required_TRAIN=8, required_RESERVED=3, qualified=False),
        not_run=['dataset annotation content audit','images','archive body reads','training',
                 'new native interventions','MOT evaluation','official TEST','Full24'],
        GPU_seconds=0, source_audit_benefit_claim=False,
        next_action='Resolve lawful label access and prospective independent cohorts; do not extend MATCH architecture yet')
    save(MASTER/'PUBLIC_RELEASE_METADATA_AUDIT.json', result)
    report = """# WWW / JEV 公共发布资料核对

这轮完成公开代码、发布页面和聚合元数据审计。没有读取新数据集的标注内容、图像、压缩包正文或官方 TEST；没有训练、原生干预或 GPU 实验。源码事实在发布审计协议前已经部分发现，因此不把本轮称为性能预注册实验。

| 来源 | 已核实的发布事实 | 对下一步的影响 |
|---|---|---|
| MMPTRACK | 官方给出 TRAIN/VAL/TEST 参与者数量及研究用途的签署访问流程 | 若有合法副本，优先核查 TRAIN/VAL；还需检查环境内 ID 和跨环境 retrieval ID 的范围 |
| WILDTRACK | GT 为 400 帧、2 Hz；释放图像为 10 Hz；原始视频为 60 Hz | 不插值增加严格监督，不把七相机视为七个独立录像；官方完整包 HEAD 大小 6,807,496,358 字节，本轮未下载正文 |
| SCOUT | 发布公告仅对 sequence_1 前八相机提供完整标注，其余是自动伪标签；网页转换版本元数据为 10 Hz | 完整标注范围不能自动认证为独立监督；转换版本频率不等于原始归档时间戳 |

UMPN 的 SCOUT 示例实际在名为 `scoutval` 的 sequence_1 相机 0–7 上训练，再做 90/10 时间切分；WILDTRACK 示例也做时间切分。仓库命名不能证明参与者或录像独立。这里没有读取真实身份标签，不能据此宣称已经发现人员重叠。窗口边界是否交叉也仍需实际接口审计。

另一个不能照搬的流程是缺失归一化统计时的全场景扫描。新适配器必须仅在冻结 TRAIN 拟合统计量，并把标注世界坐标、GT 补检测等信息留在离线监督侧。SCOUT 官方 COCO 读取器将 category_id 当作 track_id；必须核验真实身份字段、逐相机时间戳和缺失标注，不能采用通用单类别 COCO 假设。

下一范围需要合法标签来源和明确的独立 cohort。MMPTRACK 优先，但本地合法副本尚未确定。WILDTRACK 和 SCOUT 的公开页面不能直接保证需要的 8 TRAIN / 3 RESERVED 独立安全组。若后续考虑跨数据集 TRAIN/RESERVED，必须先确认官方角色、授权、标注质量及身份依赖，完整冻结后才能读取和采样；这只是设计候选，不是合格结论。

现有安全监督仍为 3 TRAIN / 1 RESERVED，学习资格不变。公开资料审计完成不证明 JEV 提升，也不完成 MASTER GOAL。

详细来源、固定提交、SHA256、逐项断言位置和未执行项见 `reports/JEV_MASTER/PUBLIC_RELEASE_METADATA_AUDIT.json`。第三方源码和网页快照仅留本地，总计 {size} 字节；GitHub 只提交审计脚本、协议、摘要和结果。

官方来源：[MMPTRACK](https://iccv2021-mmp.github.io/subpage/dataset.html)、[WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)、[SCOUT](https://scout.epfl.ch/)、[SCOUT 配置](https://github.com/cvlab-epfl/UMPN/blob/0f86307f49314a342dcaa7179647f7f126362158/configs/files/config_scout.yaml)、[分割函数](https://github.com/cvlab-epfl/UMPN/blob/0f86307f49314a342dcaa7179647f7f126362158/datasets/utils.py)、[场景加载器](https://github.com/cvlab-epfl/UMPN/blob/0f86307f49314a342dcaa7179647f7f126362158/datasets/sceneset.py)。
""".replace('{size}', str(size))
    (ROOT/'docs/WWW_JEV_PUBLIC_RELEASE_METADATA_AUDIT.md').write_text(report)
    print(json.dumps(dict(status=result['status'], assertions=len(checks), snapshot_bytes=size)))


if __name__ == '__main__':
    main()
