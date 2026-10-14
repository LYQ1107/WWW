import unittest
from unittest.mock import patch
from types import SimpleNamespace
import run_online
import run_native_prior


class Tests(unittest.TestCase):
    def test_requested_seed_reaches_loading_loop_and_provenance_and_hooks_restore(self):
        observed={};model=SimpleNamespace(geo_runtime=SimpleNamespace(active=False))
        def load(*args,**kwargs):observed['load_seed']=kwargs['seed'];return None,model
        def seed(value):observed['loop_seed']=value
        def provenance(seed):return dict(seed=seed)
        def interrupted():
            run_online.load_gmt(geo_opts={});run_online.seed_all(20261014)
            observed['report']=run_online.provenance()
            raise RuntimeError('fixture stop')
        command=['driver','--inference-seed','20261015','--video-id','1']
        with patch.object(run_online,'load_gmt',load),patch.object(run_online,'seed_all',seed),patch.object(run_online,'provenance',provenance),patch.object(run_online,'main',interrupted),patch.object(run_native_prior.sys,'argv',command):
            with self.assertRaisesRegex(RuntimeError,'fixture stop'):run_native_prior.main()
            self.assertIs(run_online.load_gmt,load);self.assertIs(run_online.seed_all,seed)
            self.assertEqual(run_native_prior.sys.argv,command)
        self.assertEqual(observed['load_seed'],20261015)
        self.assertEqual(observed['loop_seed'],20261015)
        self.assertEqual(observed['report']['seed'],20261015)
        self.assertEqual(observed['report']['command'],command)


if __name__=='__main__':unittest.main()
