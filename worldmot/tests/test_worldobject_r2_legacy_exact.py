"""Source-bound synthetic equality, not a substitute for real R2 replay."""
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
from worldobject_synthetic import obs
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
from worldmot.tracking.world_object_config import WorldObjectConfig


def test_legacy_r2_source_exact_decisions_head_inputs_and_state(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    archive=subprocess.run(['git','archive','b448af4a7dfdf738aeab0c9dd5bc01e9a875beac','worldmot/worldmot'],
        cwd=repo,check=True,stdout=subprocess.PIPE).stdout
    package=tmp_path/'frozen_r2_package';package.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as f:
        for item in f.getmembers():
            if item.isfile() and item.name.endswith('.py'):
                path=package/Path(item.name).relative_to('worldmot/worldmot')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(f.extractfile(item).read())
    spec=importlib.util.spec_from_file_location('frozen_r2_package',package/'__init__.py',
                                               submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    oldcore=importlib.import_module('frozen_r2_package.runtime.world_object_tracking_core').WorldObjectTrackingCore
    oldcfg=importlib.import_module('frozen_r2_package.tracking.world_object_config').WorldObjectConfig
    oldinputs=importlib.import_module('frozen_r2_package.models.world_object_model_inputs').model_inputs
    from worldmot.models.world_object_model_inputs import model_inputs
    new=WorldObjectTrackingCore(WorldObjectConfig(3));old=oldcore(oldcfg(3));new.reset(1);old.reset(1)
    rng=np.random.RandomState(1729)
    for t in range(80):
        observations=[]
        for camera in (1,2):
            for person in range(3):
                if rng.rand()<.12:continue
                feature=np.eye(3)[person]+rng.normal(0,.03,3)
                observations.append(obs(camera,person,center=(person*8+t*.005,0,3),feature=feature,
                                        timestamp=t,geometry=(t%7!=0)))
        before_a=model_inputs(new.bank.active(),observations,new.config,t)
        before_b=oldinputs(old.bank.active(),observations,old.config,t)
        for a,b in zip(before_a,before_b):np.testing.assert_array_equal(a.numpy(),b.numpy())
        m,n=len(new.bank.active()),len(observations)
        evidence={'assignment_logits':rng.normal(1,2,(m,n)),'null_logits':rng.normal(0,1,m),
            'existence_logits':rng.normal(3,1,m),'birth_logits':rng.normal(2,1,n),
            'update_gate':rng.uniform(.1,1,(m,n)),'memory_delta':rng.normal(0,.1,(m,n,3)),
            'birth_pair_logits':rng.normal(1,2,(n,n))}
        assert new.step(observations,t,evidence=evidence)==old.step(observations,t,evidence=evidence)
        for gid in new.bank.objects:
            a,b=new.bank.objects[gid],old.bank.objects[gid]
            for name in ('center','covariance','identity_embedding','support_points'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            assert (a.status,a.missed,a.last_timestamp)==(b.status,b.missed,b.last_timestamp)
