import unittest
from summarize_resource_samples import summarize


class Tests(unittest.TestCase):
    def fixture(self):
        manifest = dict(status='FINISHED_EXISTING_JOBS', gpu_ids=[0, 4], samples=2, interval_seconds=2.)
        devices = [dict(monotonic_seconds=str(stamp), utc=str(stamp), gpu=str(gpu), gpu_utilization_percent='50',
                        device_memory_mib='100', unattributed_compute_pids='0') for stamp in (10, 12) for gpu in (0, 4)]
        processes = [dict(monotonic_seconds='10', utc='10', pid='100', start_ticks='99', tag='job', role='run_online.py',
                          gpu=str(gpu), nvml_memory_mib='80', host_rss_kib='1024') for gpu in (0, 4)]
        return manifest, devices, processes

    def test_multigpu_process_rss_and_occurrences_are_not_double_counted(self):
        result = summarize(*self.fixture())
        process = result['process_rows'][0]
        self.assertEqual(process['sampled_host_rss_max_kib'], 1024)
        self.assertEqual(process['samples'], 1)
        self.assertEqual(process['sampled_nvml_memory_max_mib_by_gpu'], {'0': 80, '4': 80})
        self.assertTrue(process['present_at_first_probe_sample'])
        self.assertEqual(result['maximum_sample_gap_seconds'], 2.)

    def test_live_probe_or_incomplete_device_group_is_rejected(self):
        manifest, devices, processes = self.fixture()
        manifest['status'] = 'SAMPLING'
        with self.assertRaises(ValueError): summarize(manifest, devices, processes)
        manifest['status'] = 'FINISHED_EXISTING_JOBS'
        with self.assertRaises(ValueError): summarize(manifest, devices[:-1], processes)


if __name__ == '__main__':
    unittest.main()
