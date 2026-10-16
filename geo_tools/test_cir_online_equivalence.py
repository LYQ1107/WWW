"""Bounded image loader contracts; real equivalence is a separate GPU artifact."""
from unittest.mock import patch
import unittest
from run_cir_online import CurrentImages


class ImageLoaderContracts(unittest.TestCase):
    def rows(self):
        return [dict(frame_id=f,view_id=v,image_id=10*v+f) for v in (1,2) for f in (1,2)]
    def mapper(self,cfg,rows):return [dict(r,image='actual-current-image-placeholder-in-unit-only') for r in rows]
    def test_metadata_does_not_load_images(self):
        with patch('run_cir_online.map_records',side_effect=self.mapper) as mapper:
            batch=CurrentImages(None,self.rows())
            self.assertEqual(len(batch),4)
            self.assertEqual([r['image_id'] for r in batch],[11,12,21,22])
            self.assertNotIn('image',batch[0]);mapper.assert_not_called()
    def test_only_current_group_loaded_once(self):
        with patch('run_cir_online.map_records',side_effect=self.mapper) as mapper:
            batch=CurrentImages(None,self.rows())
            self.assertEqual(batch[0:1][0]['image_id'],11)
            self.assertEqual(batch[2:3][0]['image_id'],21)
            self.assertEqual(mapper.call_count,1)
            self.assertEqual(batch.processed,[11,21])
            self.assertEqual([r['image_id'] for r in mapper.call_args[0][1]],[11,21])
    def test_no_future_mixed_slice(self):
        with patch('run_cir_online.map_records',side_effect=self.mapper):
            with self.assertRaises(ValueError):CurrentImages(None,self.rows())[0:2]
    def test_no_repeated_image(self):
        with patch('run_cir_online.map_records',side_effect=self.mapper):
            batch=CurrentImages(None,self.rows());batch[0:1]
            with self.assertRaises(ValueError):batch[0:1]
    def test_no_retrograde_loading(self):
        with patch('run_cir_online.map_records',side_effect=self.mapper):
            batch=CurrentImages(None,self.rows());batch[1:2]
            with self.assertRaises(ValueError):batch[0:1]


if __name__=='__main__':unittest.main()
