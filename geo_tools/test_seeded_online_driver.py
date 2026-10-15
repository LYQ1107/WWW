import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import run_seeded_online


class Tests(unittest.TestCase):
    def test_actual_model_and_stream_seed_and_original_command_are_preserved(self):
        calls=[];reports=[]
        load=lambda **kwargs:calls.append(('load',kwargs))
        seed=lambda value:calls.append(('seed',value))
        provenance=lambda **kwargs:dict(kwargs)
        stub=SimpleNamespace(load_gmt=load,seed_all=seed,provenance=provenance)
        def main():
            self.assertNotIn('--inference-seed',sys.argv)
            stub.load_gmt();stub.seed_all(20261014);reports.append(stub.provenance())
        stub.main=main
        command=['run_seeded_online.py','--inference-seed','20261016','--tag','example','--feature-weight','2']
        with patch.dict(sys.modules,run_online=stub),patch.object(sys,'argv',command):
            run_seeded_online.main();self.assertIs(sys.argv,command)
        self.assertEqual(calls,[('load',{'seed':20261016}),('seed',20261016)])
        self.assertEqual(reports[0]['seed'],20261016);self.assertEqual(reports[0]['command'],command)
        self.assertIs(stub.load_gmt,load);self.assertIs(stub.seed_all,seed);self.assertIs(stub.provenance,provenance)


if __name__=='__main__':unittest.main()
