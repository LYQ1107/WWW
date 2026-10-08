"""Adversarial checks for irreversible exact-path cleanup guards."""
import tempfile, unittest
from pathlib import Path
from audit_jev_phase8_storage import OUT, sha, save
from safe_cleanup_jev_phase8 import B2, PROOFS, execute, validate

class Guards(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=OUT/'bootstrap');self.root=Path(self.temp.name)
        self.path=self.root/'orphan_candidate.pth';self.path.write_bytes(b'cleanup guard fixture')
        self.copy=self.root/'remaining_copy.pth';self.copy.write_bytes(self.path.read_bytes())
        self.entry={'path':str(self.path),'classification':'C_VERIFIED_ORPHAN','proofs':dict.fromkeys(PROOFS,True),'source_commit':'fixture_known_origin','reason':'test-only redundant file','verified_replica':str(self.copy),'size_bytes':self.path.stat().st_size,'sha256':sha(self.path)}
    def tearDown(self):self.temp.cleanup()
    def check(self,**kw):return validate(self.entry,[self.root],kw.get('corpus',[]),kw.get('opened',{}),kw.get('uncertain',[]))
    def test_verified_replica_required(self):
        self.copy.write_bytes(b'different')
        with self.assertRaises(AssertionError):self.check()
    def test_changed_content_rejected(self):
        self.path.write_bytes(b'changed after manifest')
        with self.assertRaises(AssertionError):self.check()
    def test_active_open_file_rejected(self):
        with self.assertRaises(AssertionError):self.check(opened={str(self.path):[123]})
    def test_new_dependency_rejected(self):
        with self.assertRaises(AssertionError):self.check(corpus=[('retained_report.md',str(self.path))])
    def test_symlink_rejected(self):
        self.path.unlink();self.path.symlink_to(self.copy)
        with self.assertRaises(AssertionError):self.check()
    def test_missing_proof_rejected(self):
        self.entry['proofs']['no_resume_dependency']=False
        with self.assertRaises(AssertionError):self.check()
    def test_protected_anchor_rejected(self):
        self.entry['path']=str(B2)
        with self.assertRaises(AssertionError):validate(self.entry,[B2.parent],[],{},[])
    def test_default_dry_run_and_empty_execution(self):
        m={'authorized_scope_roots':[str(self.root)],'entries':[]}
        self.assertTrue(execute(m)['dry_run']);self.assertEqual(execute(m,False)['freed_bytes'],0)
        self.assertTrue(self.path.exists())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    save(OUT/'bootstrap/reports/CLEANUP_GUARD_TESTS.json',{'status':'PASS'if result.wasSuccessful()else'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'old_scientific_artifacts_modified':False})
    raise SystemExit(0 if result.wasSuccessful()else 1)
