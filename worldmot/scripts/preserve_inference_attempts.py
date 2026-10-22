"""Keep interrupted prediction streams and reject incompatible resume skips."""
import json
import time
from pathlib import Path


def prepare_scene(directory, signature):
    directory = Path(directory)
    manifest = directory / 'manifest.json'
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get('status') == 'COMPLETE':
            for key, value in signature.items():
                if saved.get(key) != value:
                    raise RuntimeError('completed prediction has a different ' + key + ': ' + str(directory))
            return False
    if directory.exists() and any(directory.iterdir()):
        archive = directory.parent / '.interrupted_attempts'
        archive.mkdir(exist_ok=True)
        directory.rename(archive / (directory.name + '_' + str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)
    return True
