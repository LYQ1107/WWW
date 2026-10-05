"""Run the strict same-GPU OFF equivalence gate.

This wrapper exists so the gate can be resumed or audited independently of
the long final pipeline.  It intentionally delegates to the pipeline's
fresh-output implementation and never reuses the historical cross-GPU
comparison as formal evidence.
"""

from __future__ import annotations

from run_final_v2_pipeline import FinalPipeline


def main() -> None:
    pipeline = FinalPipeline()
    try:
        pipeline.ensure_checkpoint()
        report = pipeline.ensure_same_gpu_off_gate()
        print(report)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
