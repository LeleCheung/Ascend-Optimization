"""异步CANN包装资源生命周期检查；不需要NPU。"""
import unittest
from cann_async import Lifetimes


class Event:
    def __init__(self,done=False,error=False):self.done=done;self.error=error
    def query(self):
        if self.error:raise RuntimeError('query failed')
        return self.done
    def synchronize(self):self.done=True


class Tests(unittest.TestCase):
    def test_pending_keeps_objects_and_pointers(self):
        released=[];queue=Lifetimes(released.append);objects=object()
        queue.retain(Event(),[1,2],objects);queue.collect()
        self.assertEqual(released,[]);self.assertIs(queue.pending[0][2],objects)

    def test_out_of_order_stream_completion(self):
        released=[];queue=Lifetimes(released.append)
        queue.retain(Event(),[1],None);queue.retain(Event(True),[2],None)
        queue.collect();self.assertEqual(released,[2]);self.assertEqual(len(queue.pending),1)
        queue.finish();self.assertEqual(released,[2,1]);self.assertFalse(queue.pending)

    def test_query_error_retains_unprocessed_resources(self):
        released=[];queue=Lifetimes(released.append)
        queue.retain(Event(),[1],None);queue.retain(Event(error=True),[2],None);queue.retain(Event(True),[3],None)
        with self.assertRaises(RuntimeError):queue.collect()
        self.assertEqual(released,[]);self.assertEqual([e[1] for e in queue.pending],[[1],[2],[3]])


if __name__=='__main__':unittest.main()
