"""Refresh exact compact evidence and push a completed milestone without binaries."""
import argparse
import fcntl
import signal
from jev_oneweek_common import *


def summarize_day1():
    restores=[read(REPORTS/f'ORIGINAL_GMT_RESTORE_VIDEO{v:02d}.json') for v in DEV]
    assert all(r['raw_fields_and_global_IDs_exact'] for r in restores)
    source=read(ROOT/'reports/JEV_PHASE12/MATCH_VALIDATION_RESULTS.json')
    old=next(c for c in source['cases'] if c['case']['kind']=='off')
    matlab_source=ROOT/'reports/JEV_PHASE13/OFFICIAL_MATLAB_CROSSVIEW.json'
    matlab=next(c for c in read(matlab_source)['cases'] if c['case']=='original_GMT')
    metrics=dict(old['pooled_metrics']['strict_online']) if 'strict_online' in old['pooled_metrics'] else dict(old['pooled_metrics'])
    matlab_metrics=matlab['reports']['raw_predictions']
    for key in ['CVIDF1','CVMA']:metrics[key]=matlab_metrics.get(key)
    rows={f'M{i}':dict(status='NOT_RUN',metrics=None,reason='New unified comparison execution pending') for i in range(12)}
    rows['M0']=dict(status='REUSE_VALID_AFTER_FULL_NATIVE_PARITY',metrics=metrics,
                    exact_replayed_videos=DEV,raw_predictions_exact=True,
                    checkpoint=restores[0]['source_checkpoint'],
                    raw_metric_source=ref(ROOT/'reports/JEV_PHASE12/MATCH_VALIDATION_RESULTS.json'),
                    MATLAB_source=ref(matlab_source),same_common_Stage1_frontend=False,
                    scope='Original full GMT reference; original trained detector/VFCE/RPCE1152, separate from common VFCE1024 family')
    save(REPORTS/'MAIN_RESULTS.json',dict(status='PARTIAL_ORIGINAL_GMT_REFERENCE_ESTABLISHED',
        rows=rows,old_different_phase_scores_not_spliced=True,new_JEV_value='NOT_ESTABLISHED'))
    registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
    registry['jobs']['original_GMT_restore'].update(status='COMPLETE',results=[ref(REPORTS/f'ORIGINAL_GMT_RESTORE_VIDEO{v:02d}.json') for v in DEV])
    registry['status']='DAY1_COMPLETE_NEXT_NEW_MODEL_AND_SUPERVISION'
    save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)
    text='''# 原始 GMT 基线恢复与公平边界

原始 GMT 的完整1152D感知缓存、Stage2 model_20000 权重和原生 Gallery/Bank/Hungarian 仍在。已真实冷启动完整 DEV17/18/19（1200/1029/1052场景帧），逐个预测字段、BBox、Global ID 与冻结 Phase XII 严格在线输出完全一致。没有使用未来短轨过滤作为主结果，也没有把点式共同缓存 GTA 当作完整 GMT。

因此可以在明确原前端条件下复用原始真实 pooled HOTA78.110/AssA80.821/IDF1 94.814/IDSW204/MOTA91.547。官方 MATLAB 的原始 CVIDF1/CVMA 来源单独 SHA 绑定。它使用已训练原Stage2前端，不能与共同Stage1 VFCE1024系列当成单一模块因果差值。新同证据 MLP/Attention 等仍须运行，不拿旧失败MLP填表。

本次恢复是缓存感知下真实变异状态在线跟踪，包含重新评分、求解和真实提交，不是缓存未来ID或动作。缓存壁钟时间不属于完整图像 scene FPS。完整视频三次重启效率仍待执行。一次观察器字段名工程错误已修复，原错误日志、START 和失败 SHA 保留。
'''
    (DOCS/'BASELINE_PARITY.md').write_text(text)


def snapshot(message=None):
    protect()
    if (REPORTS/'FORMAL_CURRENT_RESULTS.json').exists():
        from jev_oneweek_reporting import refresh
        refresh()
    failures=[read(p) for p in (OUT/'failures').glob('*.json')]
    save(REPORTS/'ENGINEERING_FAILURES.json',dict(status='RETAINED',failures=failures,
         publication_failures=[read(p) for p in (OUT/'publication_failures').glob('*.json')],
         no_scientific_failure_hidden=True))
    paths=[p for folder in (DOCS,REPORTS,ROOT/'gtr/modeling/jev_oneweek',ROOT/'configs/JEV_ONEWEEK') if folder.exists()
           for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name!='ARTIFACT_SHA_MANIFEST.json']
    paths+=list((ROOT/'reproduction_tools').glob('jev_oneweek_*.py'))
    assert all(p.stat().st_size<8*2**20 for p in paths),'Only compact publication files'
    for p in (ROOT/'reproduction_tools').glob('jev_oneweek_*.py'):compile(p.read_text(),str(p),'exec')
    save(REPORTS/'ARTIFACT_SHA_MANIFEST.json',dict(status='VERIFIED_COMPACT_MILESTONE',
         no_binary_upload=True,science_status='NOT_ESTABLISHED',files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    if message is None:return
    selected=[str(p.relative_to(ROOT)) for p in paths]+['reports/JEV_ONEWEEK/ARTIFACT_SHA_MANIFEST.json']
    subprocess.run(['git','add',*selected],cwd=ROOT,check=True)
    subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check'],cwd=ROOT,check=True)
    unchanged=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
    assert unchanged in [0,1]
    if unchanged:
        subprocess.run(['git','-c','gc.auto=0','commit','-m',message],cwd=ROOT,check=True)
    else:print('NO_NEW_COMMIT_RETRY_EXISTING_HEAD',flush=True)


def network_command(command,env,timeout):
    # Killing git alone leaves its ssh/proxy children alive. Isolate this
    # publication subprocess group so timeouts can reap only our own children.
    process=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:stdout,stderr=process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate()
        raise subprocess.TimeoutExpired(command,timeout,output=stdout,stderr=stderr)
    if stderr:print(stderr,end='',flush=True)
    if process.returncode:raise subprocess.CalledProcessError(process.returncode,command,output=stdout,stderr=stderr)
    return stdout


def _publish_locked(message=None):
    with (OUT/'REGISTRY.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        snapshot(message)
    if message is None:return
    ssh="ssh -i /home/liuyeqiang/.ssh/id_ed25519_www_codex -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o HostKeyAlias=github.com -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p'"
    env=dict(os.environ,GIT_SSH_COMMAND=ssh)
    for attempt in range(3):
        try:
            network_command(['git','push','origin','HEAD:refs/heads/jev/www-jev-oneweek-20261012'],env,90)
            local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            remote=network_command(['git','ls-remote','origin','refs/heads/jev/www-jev-oneweek-20261012'],env,45).split()[0]
            assert local==remote
            break
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
            save(OUT/'publication_failures'/f'{time.time_ns()}.json',dict(status='PUBLICATION_ATTEMPT_FAILED',attempt=attempt,
                 command=error.cmd,error=str(error),local_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
            if attempt==2:raise
            time.sleep(3)
    print('VERIFIED_REMOTE_HEAD',local,flush=True)


def publish(message=None):
    with (OUT/'PUBLICATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _publish_locked(message)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--day1',action='store_true');p.add_argument('--commit-message');a=p.parse_args()
    if a.day1:summarize_day1()
    publish(a.commit_message)
