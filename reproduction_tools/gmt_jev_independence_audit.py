"""Current independent-validation qualification from repository history only."""
from gmt_jev_common import *

def main():
 paths=['reports/JEV_PHASE19/HISTORICAL_EXPOSURE_AUDIT.json',
  'reports/JEV_PHASE19/INDEPENDENCE_QUALIFICATION_PROTOCOL.json',
  'reports/JEV_PHASE13/SPLIT_LEAKAGE_AUDIT.json',
  'reports/JEV_MASTER/DATA_ACCESS_USER_CONFIRMATION.json',
  'reports/JEV_MASTER/EXISTING_SOURCE_ACCESS_AUDIT.json']
 history=read(ROOT/paths[0]);access=read(ROOT/paths[3]);existing=read(ROOT/paths[4]);goal=read(REPORTS/'MASTER_GOAL.json')
 result=dict(status='NO_QUALIFIED_INDEPENDENT_SOURCE_ESTABLISHED_WITHIN_CURRENT_SCOPE',
  evidence=[ref(ROOT/p) for p in paths],current_scope=goal['scope'],
  controller_exposure=history['documented_previous_controller_exposure'],
  backbone_exposure=history['stage1'],
  exposure_interpretation='Backbone exposure does not by itself decide controller-heldout qualification, but it prevents a whole-model image-unseen claim. Prior controller evaluation and unknown provenance require separate disclosure.',
  historical_reserved=history['this_round_reserved'],reserved_claim=history['reserved_claim'],
  unresolved_08_15=history['other_new_scope_exposure'],
  external_copy_user_reply=dict(date=access['client_date'],answer=access['answer'],scope=access['scope'],
   current_global_absence_asserted=False),
  existing_external_copy=dict(status=existing['status'],limitation=existing['limitation']),
  qualification_requirements=['Lawful access and current scope authorization','Backbone and controller training/selection exposure disclosed',
   'Video/person/scene separation and historical outcome exposure audited','Freeze all methods/seeds/evaluators before new outcomes',
   'Matched full original GMT cold runs on the same complete videos','Do not call previously exposed validation fresh independent test'],
  pixels_or_annotations_read_this_audit=False,new_download=False,new_evaluation=False,
  do_not_unseal=True,scientific_success=False,
  scope_note='This audit is not a scope expansion, data access request, validation run, or proof that no suitable dataset exists anywhere.',source=binding())
 save(REPORTS/'INDEPENDENT_VALIDATION_QUALIFICATION.json',result)
 (DOCS/'INDEPENDENT_VALIDATION_QUALIFICATION_ZH.md').write_text('''# 当前独立验证资格审计

仅复核已存项目记录，未打开新视频、图像、标注或下载数据。当前获准训练12/13/14、开发16；16已暴露，不能作为独立验证。

Phase19 记录了2/3/5的旧controller-heldout评测、4的调试、6的验证/校准、7的学习/因果标签以及9/10/11的旧controller-heldout评测。此前 reserved10/11/15 是该轮前瞻控制器验证范围，10/11有历史评测；8/15的从未使用状态没有建立，不能凭缺少直接训练条目就宣称新独立数据。

历史 backbone Stage1 已见全部 TRAIN24。这不自动否定单独冻结的 controller-heldout 问题，但不能声称整模型未见图像，且不能把旧评测范围重新命名为 fresh test。新的资格标准和全部暴露必须在后续结果前固定。

用户2026-10-12答复没有 MMPTRACK/SCOUT 合法副本；该答复只是当时状态，不证明现在或全服务器绝对不存在副本。此前找到的 WILDTRACK 是重复使用的 TRAIN 副本，旧记录明确不提供独立 RESERVED 监督。以上均按源文件SHA引用，不读取原图或复核标签。

结论：当前范围尚未建立合格独立完整视频验证。需要合法来源、授权、pretrain/controller暴露与选择历史、场景/人物关系、固定方法/种子/评测协议，再做 matched 完整 GMT 的冷启动对照。该审计不增加使用范围、不解封20/21/22、不读TEST，也不会因开发指标正向就改变科学成功状态。已有计时与TRAIN诊断继续执行。
''')
 print('INDEPENDENCE_AUDIT',result['status'],flush=True)

if __name__=='__main__':main()
