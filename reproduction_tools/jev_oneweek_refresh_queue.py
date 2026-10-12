"""Finite publication refresh when completed primary evidence changes."""
from jev_oneweek_common import *
from jev_oneweek_publish import publish


def main():
    last=None
    while True:
        paths=[]
        for pattern in ['MATLAB_*_formal_*.json','EFFICIENCY_*_formal_*.json','COMPONENT_PROFILE_*.json',
                        'TYPED_QUESTION_ORDINARY_CONTROL_COMPARISON.json','ORIGINAL_GMT_STAGE2_COMPONENT_PROFILE.json']:
            paths.extend(REPORTS.glob(pattern))
        signature=tuple((p.name,sha(p)) for p in sorted(paths))
        if signature!=last:
            publish('Refresh completed pooled main tables reviewer evidence and true efficiency measurements')
            last=signature
            save(OUT/'publication_refresh/PROGRESS.json',dict(status='ACTIVE_UNTIL_FINITE_SPRINT_FINISHED',completed_primary_artifacts=len(paths)))
        if (OUT/'FINAL_AUDIT_READY.json').exists():break
        time.sleep(30)
    print('FINITE_PUBLICATION_REFRESH_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('publication_refresh',traceback.format_exc());raise
