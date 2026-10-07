from pathlib import Path
import json,os,sys
HERE=Path(__file__).resolve().parent
SETTINGS=json.loads((HERE/"mps_settings.json").read_text())
sys.path.insert(0,SETTINGS["source_root"]+"/reproduction_tools")
import run_segmented_small_gate as entry
original_environment=entry.environment

def environment(gpu):
    env=original_environment(gpu)
    env["CUDA_VISIBLE_DEVICES"]=SETTINGS["gpu_uuids"][str(gpu)]
    env["CUDA_MPS_PIPE_DIRECTORY"]=SETTINGS["pipe"]
    env["CUDA_MPS_LOG_DIRECTORY"]=SETTINGS["logs"]
    return env
entry.environment=environment
entry.__file__=str(Path(__file__).resolve())
original_engine=entry.engine

def engine():
    selected=os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected in SETTINGS["gpu_uuids"]:
        os.environ["CUDA_VISIBLE_DEVICES"]=SETTINGS["gpu_uuids"][selected]
    os.environ["CUDA_MPS_PIPE_DIRECTORY"]=SETTINGS["pipe"]
    os.environ["CUDA_MPS_LOG_DIRECTORY"]=SETTINGS["logs"]
    return original_engine()
entry.engine=engine
if __name__=="__main__":entry.main()
