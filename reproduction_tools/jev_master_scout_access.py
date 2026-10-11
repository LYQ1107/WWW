"""Read public download UI source; never submit registration or download data."""
from pathlib import Path
import subprocess
from jev_master_audit_common import ROOT, MASTER, protect, read, ref, save, sha


def main():
    protect()
    protocol=read(MASTER/'SCOUT_ACCESS_TERMS_PROTOCOL.json')
    path=Path(protocol['snapshot']['path'])
    assert sha(path)==protocol['snapshot']['SHA256']
    assert path.stat().st_size==21088
    source=path.read_text()
    home=Path(protocol['route_binding']['path'])
    assert sha(home)==protocol['route_binding']['SHA256']
    assert './DownloadView-BzwziPJH.js' in home.read_text()
    checks=[]
    assertions={
        'full_name': 'Researcher name is collected by the normal registration flow',
        'institution': 'Institution is collected',
        'intended_use': 'Research purpose is collected',
        'agree_to_terms': 'Terms acceptance is required by the UI',
        '/api/registrations/': 'Registration uses a POST request',
        '/api/registrations/verify-email/': 'The normal download flow verifies the registered email',
        '/api/downloads/log-download/': 'Download authorization is requested before creating the archive link',
        'user_request:f.value.id': 'Download request identifies the registration record',
        'method:"POST"': 'The UI performs external mutations; this audit executes none of them',
        'Dataset License Agreement': 'Dataset-specific terms are in the lazy download component',
        '<strong>No Redistribution:</strong>': 'Dataset portions require permission for redistribution',
        'academic and research purposes only': 'The registration acknowledgement limits intended use',
        'h(839.92)': 'Annotation package size is a UI advertisement, not measured archive bytes',
        'scout_annotations.tar.gz': 'The advertised standalone annotation package is a tar.gz',
    }
    for marker,meaning in assertions.items():
        assert marker in source,marker
        checks.append(dict(byte_offset=len(source[:source.index(marker)].encode()),meaning=meaning))
    result=dict(status='COMPLETE_PUBLIC_DOWNLOAD_FLOW_AUDIT',
        actor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        actor=ref(Path(__file__)),protocol=ref(MASTER/'SCOUT_ACCESS_TERMS_PROTOCOL.json'),
        official_page='https://scout.epfl.ch/download',source=protocol['snapshot'],
        source_url=protocol['source_url'],route_binding=protocol['route_binding'],assertions=checks,
        normal_access_flow=['Researcher registration with name/institution/email/purpose and terms',
                            'Registered email verification','Download authorization/logging',
                            'Archive link generated after successful authorization'],
        dataset_terms=dict(attribution_required=True,redistribution_requires_explicit_permission=True,
                           research_use_acknowledgement=True,privacy_ethics_conditions=True,
                           provided_without_warranty=True,registration_data_storage_disclosed=True,
                           classification='Dataset-specific agreement; not inferred from JS dependency MIT licenses'),
        advertised_annotations_package=dict(value=839.92,unit='MB as displayed by UI',
                                            actual_transport_bytes='NOT_MEASURED',
                                            expanded_bytes='NOT_MEASURED',exceeds_64_MiB_audit_budget=True),
        existing_budget_MiB=64,
        classification='NORMAL_DOWNLOAD_ACCESS_AND_BUDGET_NOT_ESTABLISHED_FOR_AGENT',
        backend_authorization_live_behavior='NOT_TESTED; static UI flow is evidence, not a claim about exploitability',
        normal_role_audit='No explicit independent participant/capture TRAIN/RESERVED manifest in this component',
        new_source_data_eligible=False,existing_learning_gate_unchanged=True,
        no_external_registration_submitted=True,no_email_verification_submitted=True,
        no_download_authorization_submitted=True,no_protected_archive_or_viewer_GT_requested=True,
        source_snapshot_bytes=path.stat().st_size,new_GT_bytes=0,new_image_bytes=0,
        GPU_seconds=0,training=False,native_actions=0,
        lawful_next_route='User-provided existing approved TRAIN/VAL copy or a normally authorized manual subset. Re-freeze actual source roles/quality/dependencies before reading labels; no inference from filenames alone.')
    save(MASTER/'SCOUT_ACCESS_TERMS_AUDIT.json',result)
    (ROOT/'docs/WWW_JEV_SCOUT_ACCESS_AUDIT.md').write_text('''# SCOUT 访问与标注预算核查

本轮读取了官方 `/download` 页面引用的独立JavaScript代码块，21,088字节，SHA256和首页路由绑定已冻结。上轮只检查了首页代码，因此当时的“数据许可未验证”仍是正确的历史状态。本轮补充核实了下载页的数据专属使用条款和正常访问流程。

下载流程要求研究者填写姓名、机构、邮箱、研究用途并接受条款；随后验证登记邮箱，向下载授权接口提交登记记录，再生成归档链接。这是静态源码审计，没有提交任何登记、邮箱验证、授权请求或联系数据提供者，也未测试后端是否存在别的访问方式。

条款要求引用论文，未经明确许可不得重新分发数据或其部分，并包含科研用途、隐私伦理、无保证以及登记信息存储说明。这是数据专属协议，不能用首页Vue/Three.js的MIT许可代替。GitHub继续只保存我们自己的代码、哈希和摘要。

页面标注包标称839.92MB，超过64MiB的A阶段预算。这个数字来自网页显示，不是HTTP实测大小，也不是展开大小；本轮没有请求归档正文或受保护的下载接口。组件没有提供可证明人物/采集独立的TRAIN/RESERVED清单。

因此不能把SCOUT称为已获得、预算内、可直接训练或独立验证的数据。下一步需要获准使用的本地副本或正常授权的小量人工标注子集，再冻结实际角色、标签质量和依赖关系。当前已定位的WILDTRACK前320TRAIN仍是历史暴露的一次连续采集，未提供新的RESERVED；原3 TRAIN/1 RESERVED、8＋3学习资格保持不变。

已向用户询问现有获准副本的目录路径。这是缺少数据来源信息，并非要求重复确认普通工程步骤；不把旧问卷未回复作为读取既有授权数据的阻碍。MASTER研究目标未完成。

来源：[SCOUT官方访问页面](https://scout.epfl.ch/download)，及其[固定下载组件](https://scout.epfl.ch/assets/DownloadView-BzwziPJH.js)。详细断言位置、来源SHA、实际未执行项见SCOUT_ACCESS_TERMS_AUDIT.json。
''')
    print('COMPLETE_SCOUT_ACCESS_SOURCE_AUDIT',len(checks),'new_GT_bytes0','GPU0')


if __name__=='__main__':main()
