import unittest
import numpy as np
from extract_features import select, windows

class SamplingTests(unittest.TestCase):
    def test_budget_and_order(self):
        for n in (1,15,16,127,256,399):
            for b in (16,128,256):
                for method in ('uniform','motion-aware','adaptive','all'):
                    idx=select(np.random.default_rng(n).random(n),b,method)
                    self.assertEqual(len(idx),n if method=='all' else min(n,b))
                    self.assertTrue(np.all(np.diff(idx)>0))
                    self.assertTrue(np.all((idx>=0)&(idx<n)))
    def test_adaptive(self):
        s=np.zeros(100); s[40:60]=10
        idx=select(s,20,'adaptive')
        self.assertIn(0,idx); self.assertIn(99,idx)
        self.assertGreaterEqual(sum((idx>=40)&(idx<60)),10)
    def test_windows(self):
        for n in (1,8,16,17,24,25,200):
            w=windows(n)
            self.assertEqual(w[0][0],0); self.assertEqual(w[-1][-1],n-1)
            self.assertTrue(all(len(x)==16 for x in w))
            self.assertTrue(all(np.all(np.diff(x)<=1) for x in w))
            self.assertTrue(all(b[0]-a[0]==8 for a,b in zip(w,w[1:])))

if __name__=='__main__': unittest.main()
