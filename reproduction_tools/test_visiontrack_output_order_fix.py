#!/usr/bin/env python3
"""Small regression test for the VisionTrack evaluator order contract."""

import unittest

from gtr.evaluation.mot_evaluation import align_visiontrack_inputs_to_outputs


class VisionTrackOutputOrderTest(unittest.TestCase):
    @staticmethod
    def record(image_id, frame, view, view_num=2):
        return {
            "image_id": image_id,
            "video_id": 7,
            "frame_id": frame,
            "view_id": view,
            "view_num": view_num,
        }

    def test_view_block_to_frame_major(self):
        inputs = [
            self.record(101, 1, 1),
            self.record(103, 2, 1),
            self.record(102, 1, 2),
            self.record(104, 2, 2),
        ]
        aligned = align_visiontrack_inputs_to_outputs(
            "VISION_test", inputs, [object()] * len(inputs)
        )
        self.assertEqual(
            [(item["image_id"], item["frame_id"], item["view_id"]) for item in aligned],
            [(101, 1, 1), (102, 1, 2), (103, 2, 1), (104, 2, 2)],
        )

    def test_identity_cases(self):
        single = [self.record(1, 1, 1, view_num=1)]
        other = [self.record(2, 1, 1, view_num=2)]
        outputs = [object()]
        self.assertIs(
            align_visiontrack_inputs_to_outputs("VISION_test", single, outputs),
            single,
        )
        self.assertIs(
            align_visiontrack_inputs_to_outputs("OTHER", other, outputs),
            other,
        )

    def test_invalid_structure_fails_closed(self):
        inputs = [self.record(1, 1, 1), self.record(2, 1, 2)]
        with self.assertRaisesRegex(ValueError, "lengths"):
            align_visiontrack_inputs_to_outputs("VISION_test", inputs, [object()])
        with self.assertRaisesRegex(ValueError, "divisible"):
            align_visiontrack_inputs_to_outputs(
                "VISION_test", inputs + [self.record(3, 2, 1)], [object()] * 3
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
