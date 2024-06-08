import unittest
import numpy as np
import BeliefPropagation
import snippets

class TestBeliefPropagation(unittest.TestCase):

    code = snippets.n5k2_acyclic
    num_cws = 1e5
    EbN0 = 2

    def setUp(self):
        self.bp = BeliefPropagation.BeliefPropagation(adjacency_matrix=self.code.adjacency_matrix(), state_domain_size=2)
        self.rx = snippets.simulateAWGNChannelTransmission(self.code, self.EbN0, self.num_cws)
    
    def tearDown(self):
        pass
    
    def test_maxproduct_bp(self):
        # compute MPA assignment

        # compute factors

        # run bp

        # decode with beliefs

        # compare with Blockwise-MAP

        llrs = np.empty(self.rx.shape)
        iters = np.empty(self.rx.shape[0])
        for (i, y) in enumerate(self.rx):
            (llrs[i,:], iters[i]) = self.code.decode_awgn(y, self.EsN0_lin, self.spa_iters, max_product=True)
        mpa_assignment = 0.5*(1-np.sign(llrs)) # Hard decision on LLRs.
        pass

    def test_sumproduct_bp(self):

        pass