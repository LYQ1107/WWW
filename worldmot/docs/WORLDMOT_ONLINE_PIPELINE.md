# Online pipeline

`OnlineWorldMOT.reset(scene_id)` starts a new identity namespace. `step(images_by_camera, timestamp)` performs geometry, instance construction, causal alignment, memory association, and projection once. It does not read future images or annotations, and it does not reset memory between timestamps.

`scripts/run_pipeline.py` writes one JSON object per timestamp and a summary. A run is scientifically useful only when the summary includes the actual geometry status and alignment diagnostics for every attempted pair.
