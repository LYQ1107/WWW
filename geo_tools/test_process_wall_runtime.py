import unittest
from audit_process_wall_runtime import process_cost, affected_windows, superseded_child_windows


class Tests(unittest.TestCase):
    def test_only_confirmed_exited_superseded_child_has_missing_outer_cost(self):
        stage=dict(status='RUNNING',pid=100)
        window=dict(path='handoff.json',controller_pid=5,existing_children={'100':'ticks'},
                    observed_status='ORIGINAL_DISPATCHER_SUPERSEDED_AFTER_CHILD_EXIT')
        self.assertEqual(superseded_child_windows(5,stage,[window]),['handoff.json'])
        self.assertEqual(superseded_child_windows(6,stage,[window]),[])
        self.assertEqual(superseded_child_windows(5,stage,[dict(window,observed_status='REPLACEMENT_QUEUES_RUNNING')]),[])
        self.assertEqual(superseded_child_windows(5,dict(stage,pid=101),[window]),[])

    def test_cold_export_cost_is_included_and_missing_duration_cannot_be_zero_filled(self):
        stage = dict(status='PASS', exit_code=0, seconds=12.)
        run = dict(full_video=True, used_cache=False, seconds=10., images=24)
        value = process_cost(stage, run, False)
        self.assertEqual(value['process_camera_image_fps'], 2.)
        self.assertEqual(value['outside_loop_seconds'], 2.)
        with self.assertRaises(ValueError):
            process_cost(dict(stage, seconds=9.), run, False)
        with self.assertRaises(ValueError):
            process_cost(stage, dict(run, full_video=False), False)
        self.assertIsNone(process_cost(stage, run, True))

    def test_pause_overlap_without_recorded_child_is_excluded(self):
        stage = dict(pid=100, started_utc='2026-10-15T00:00:10+00:00', finished_utc='2026-10-15T00:00:20+00:00')
        window = dict(path='window.json', controller_pid=5, existing_children={},
                      conservative_start_timestamp=1792022415., conservative_end_timestamp=1792022418.)
        self.assertEqual(affected_windows(5, stage, [window]), ['window.json'])
        self.assertEqual(affected_windows(6, stage, [window]), [])
        later = dict(stage, started_utc='2026-10-15T00:00:30+00:00', finished_utc='2026-10-15T00:00:40+00:00')
        self.assertEqual(affected_windows(5, later, [window]), [])
        self.assertEqual(affected_windows(5, later, [dict(window, existing_children={'100': 'starttime'})]), ['window.json'])


if __name__ == '__main__':
    unittest.main()
