"""Risk head capacity/input/RNG contracts; no tracking-performance claim."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import torch
import common
from gtr.modeling.identity_revision.model import NativeRiskHead,INPUT_CONTRACT,GLOBAL_INDICES,LOCAL_INDICES,load_native_risk


class RiskHeadContracts(unittest.TestCase):
    def test_equal_actual_capacity_and_all_parameters_used(self):
        for name in ('ordinary_mlp','grouped_risk'):
            model=NativeRiskHead(name);self.assertEqual(sum(p.numel() for p in model.parameters()),961)
            y=model(torch.randn(7,12));self.assertEqual(y.shape,(7,));y.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertEqual(sorted(GLOBAL_INDICES+LOCAL_INDICES),list(range(12)))
    def test_extra_gt_column_and_nonfinite_inputs_rejected(self):
        m=NativeRiskHead('ordinary_mlp')
        for x in (torch.zeros(3,13),torch.zeros(12),torch.ones(2,12,dtype=torch.long),torch.full((1,12),float('nan'))):
            with self.assertRaises(ValueError):m(x)
        with self.assertRaises(ValueError):m({'features':torch.zeros(2,12),'GT':torch.ones(2)})
    def test_empty_queries_have_valid_zero_length_output(self):
        for name in ('ordinary_mlp','grouped_risk'):self.assertEqual(NativeRiskHead(name)(torch.zeros(0,12)).shape,(0,))
    def test_verified_load_preserves_rng_and_exact_output(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('ordinary_mlp','grouped_risk'):
                model=NativeRiskHead(name).eval();model.set_normalization(torch.arange(12).float(),torch.ones(12))
                path=Path(temp)/(name+'.pt');torch.save(dict(input_contract=INPUT_CONTRACT,architecture=name,state_dict=model.state_dict(),threshold=.25),path)
                digest=hashlib.sha256(path.read_bytes()).hexdigest();before=torch.get_rng_state().clone()
                loaded,threshold=load_native_risk(path,digest);self.assertTrue(torch.equal(before,torch.get_rng_state()))
                self.assertEqual(threshold,.25);x=torch.arange(24).float().reshape(2,12)
                self.assertTrue(torch.equal(model(x),loaded(x)));self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                with self.assertRaises(ValueError):load_native_risk(path,'wrong')
    def test_normalization_and_architecture_contracts(self):
        with self.assertRaises(ValueError):NativeRiskHead('teacher')
        model=NativeRiskHead('ordinary_mlp')
        with self.assertRaises(ValueError):model.set_normalization(torch.zeros(12),torch.zeros(12))
    def test_constructor_precision_does_not_depend_on_global_default(self):
        original=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for name in ('ordinary_mlp','grouped_risk'):
                model=NativeRiskHead(name)
                self.assertTrue(all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters()))
                self.assertEqual(model(torch.zeros(2,12,dtype=torch.float32)).dtype,torch.float32)
        finally:torch.set_default_dtype(original)


if __name__=='__main__':unittest.main()
