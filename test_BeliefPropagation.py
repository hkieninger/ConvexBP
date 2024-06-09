import unittest
import numpy as np
import BeliefPropagation
import snippets

class TestBeliefPropagation(unittest.TestCase):

    code = snippets.n5k2_acyclic
    num_cws = int(1e4)
    EbN0 = 2
    bp_iterations = 20
    convergence_threshold = 1e-6

    def setUp(self):
        self.bp = BeliefPropagation.BeliefPropagation(adjacency_matrix=self.code.adjacency_matrix(), state_domain_size=2)
        self.rx = snippets.simulateAWGNChannelTransmission(self.code, self.EbN0, self.num_cws)
    
    def tearDown(self):
        pass
    
    def test_maxproduct_bp(self):
        '''
        prueft ob das Ergebnis von MPA Blockwise-MAP entspricht, dies muss fuer einen Code ohne Zyklen erfuellt sein
        '''
        # compute MPA assignment
        mpa_estimate, iterations = snippets.decode_with_standart_bp(
            blockwise=True,
            rx=self.rx,
            EbN0=self.EbN0,
            code=self.code,
            bp=self.bp,
            max_iters=self.bp_iterations,
            convergence_threshold=1e-6)
        self.assertEqual(np.sum(iterations == self.bp_iterations), 0, "BP did not converge for some cases") # assert that BP converged

        # compute Blockwise-MAP and compare
        map_estimate = snippets.bruteforce_blockwiseMAP_AWGNChannel(self.code, self.rx)
        mpa_unequal_map = np.sum(np.logical_xor(mpa_estimate, map_estimate), axis=1) > 0
        self.assertEqual(np.sum(mpa_unequal_map), 0, "some MPA estimates do not correspond to blockwise MAP")

    def test_sumproduct_bp(self):
        '''
        prueft ob das Ergebnis von SPA Bitwise-MAP entspricht, dies muss fuer einen Code ohne Zyklen erfuellt sein
        '''
        # compute SPA assignment
        spa_estimate, iterations = snippets.decode_with_standart_bp(
            blockwise=False,
            rx=self.rx,
            EbN0=self.EbN0,
            code=self.code,
            bp=self.bp,
            max_iters=self.bp_iterations,
            convergence_threshold=1e-6)
        self.assertEqual(np.sum(iterations == self.bp_iterations), 0, "BP did not converge for some cases") # assert that BP converged

        # compute Bitwise-MAP and compare
        map_estimate = snippets.bruteforce_bitwiseMAP_AWGNChannel(self.code, self.rx, self.EbN0)
        mpa_unequal_map = np.sum(np.logical_xor(spa_estimate, map_estimate), axis=1) > 0
        self.assertEqual(np.sum(mpa_unequal_map), 0, "some SPA estimates do not correspond to bitwise MAP")