import unittest
import common
from gtr.modeling.identity_revision.output_buffer import OutputBuffer


class BufferContracts(unittest.TestCase):
    def test_zero_delay_is_immediately_immutable(self):
        buffer=OutputBuffer(0);payload={'ids':[1]};released=buffer.push(1,payload)
        payload['ids'][0]=9;self.assertEqual(released,[(1,{'ids':[1]})])
        with self.assertRaises(ValueError):buffer.revise_pending(1,{'ids':[2]})
    def test_window2_seals_in_order(self):
        buffer=OutputBuffer(2);self.assertEqual(buffer.push(1,[1]),[]);self.assertEqual(buffer.push(2,[2]),[])
        buffer.revise_pending(2,[20]);self.assertEqual(buffer.push(3,[3]),[(1,[1])])
        with self.assertRaises(ValueError):buffer.revise_pending(1,[10])
        self.assertEqual(buffer.flush(),[(2,[20]),(3,[3])]);self.assertEqual(buffer.sealed_through,3)
    def test_reset_no_crossvideo_state(self):
        buffer=OutputBuffer(4);buffer.push(100,[1]);buffer.reset();self.assertEqual(buffer.push(1,[2]),[])
        self.assertEqual(buffer.flush(),[(1,[2])])
    def test_nonmonotonic_rejected(self):
        buffer=OutputBuffer(2);buffer.push(2,[1])
        with self.assertRaises(ValueError):buffer.push(2,[2])


if __name__=='__main__':unittest.main()
