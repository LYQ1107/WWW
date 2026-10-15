"""Read actual microbenchmarks; incompatible results never become speedups."""
import argparse
import numpy as np
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    paths={variant:RUNTIME/'encoder_microbench'/args.tag/variant/'manifest.json' for variant in ['original','fast']}
    runs={variant:json.loads(path.read_text()) for variant,path in paths.items()}
    result=dict(scope='fixed TRAIN three-pair encoder/ROI/camera microbenchmark; not tracking accuracy or end-to-end throughput',
                sources={variant:dict(path=str(path),sha256=sha256(path)) for variant,path in paths.items()},rows=[],provenance=provenance())
    if any(run['status']!='PASS' for run in runs.values()):
        result.update(status='NO_VALID_SPEED_COMPARISON',failures={variant:dict(status=run['status'],error_type=run.get('error_type'),error=run.get('error')) for variant,run in runs.items()})
    else:
        if runs['original']['weights_sha256']!=runs['fast']['weights_sha256']:raise ValueError('weight mismatch')
        if runs['original'].get('camera_fp32_input',False)!=runs['fast'].get('camera_fp32_input',False):raise ValueError('camera precision bridge mismatch')
        original={(r['video_id'],r['frame_id']):r for r in runs['original']['rows']}
        fast={(r['video_id'],r['frame_id']):r for r in runs['fast']['rows']}
        if set(original)!=set(fast):raise ValueError('sample mismatch')
        for key in original:
            left,right=original[key],fast[key]
            a,b=np.load(left['artifact']),np.load(right['artifact'])
            matrices=[]
            for view in range(2):
                x,y=a[f'view{view}'].astype(np.float32),b[f'view{view}'].astype(np.float32)
                if x.shape!=y.shape:raise ValueError('ROI shape mismatch')
                matrices.append(float(np.max(np.abs(x-y))) if x.size else 0.)
            result['rows'].append(dict(video_id=key[0],frame_id=key[1],original_median_seconds=left['median_two_order_seconds'],fast_median_seconds=right['median_two_order_seconds'],
                  original_over_fast_ratio=left['median_two_order_seconds']/right['median_two_order_seconds'],target_vector_max_abs_difference=max(matrices),
                  measured_single_call_components={variant:{component:dict(p50=float(np.percentile([call[component] for call in row['measured_single_call_timings']],50)),p95=float(np.percentile([call[component] for call in row['measured_single_call_timings']],95))) for component in ['aggregator_seconds','camera_seconds','transfer_aggregator_roi_camera_cpu_result_seconds']} for variant,row in [('original',left),('fast',right)]},
                  storage_scope='FP16 saved ROI vectors; this comparison does not prove bitwise live feature parity'))
        result.update(status='PASS_MICROBENCHMARK_ONLY',feature_proxy_metrics={variant:run['feature_proxy_metrics'] for variant,run in runs.items()},
                      camera_fp32_input=runs['original'].get('camera_fp32_input',False),
                      limitations='five repeated encoder measurements on three TRAIN pairs; changed features require closed-loop tracking evaluation before integration; no long-sequence speed multiplier or full-system FPS claim')
    save_json(RUNTIME/'encoder_microbench'/args.tag/'comparison.json',result)
    save_json(ROOT/'results'/f'{args.tag}_comparison.json',result)


if __name__=='__main__':main()
