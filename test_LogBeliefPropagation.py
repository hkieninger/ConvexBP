import unittest
import numpy as np
import snippets
import LogBeliefPropagation

class TestLogBeliefPropagation(unittest.TestCase):

    code = snippets.n8k8_acyclic
    num_cws = int(1e4)
    EbN0 = 2
    bp_iterations = 20
    rtol = 1e-8
    atol = 1e-12
    infty = 1e6

    def setUp(self):
        super().setUp()
        np.random.seed(3845)
        self.bp = LogBeliefPropagation.LogBeliefPropagation(adjacency_matrix=self.code.adjacency_matrix(), state_domain_size=2)
        self.rx = snippets.simulateAWGNChannelTransmission(self.code, self.EbN0, self.num_cws)
    
    def test_log_maxproduct_bp(self):
        '''
        prueft ob das Ergebnis von MPA Blockwise-MAP entspricht, dies muss fuer einen Code ohne Zyklen erfuellt sein
        '''
        # compute MPA assignment
        mpa_estimate, iterations = snippets.decode_with_standart_log_bp(
            blockwise=True,
            rx=self.rx,
            EbN0=self.EbN0,
            code=self.code,
            bp=self.bp,
            max_iters=self.bp_iterations,
            rtol=self.rtol,
            atol=self.atol,
            infty=self.infty)
        self.assertEqual(np.sum(iterations == self.bp_iterations), 0, "BP did not converge for some cases") # assert that BP converged

        # compute Blockwise-MAP and compare
        map_estimate = snippets.bruteforce_blockwiseMAP_AWGNChannel(self.code, self.rx)
        mpa_unequal_map = np.sum(np.logical_xor(mpa_estimate, map_estimate), axis=1) > 0
        self.assertEqual(np.sum(mpa_unequal_map), 0, "some MPA estimates do not correspond to blockwise MAP")

    def test_log_sumproduct_bp(self):
        '''
        prueft ob das Ergebnis von SPA Bitwise-MAP entspricht, dies muss fuer einen Code ohne Zyklen erfuellt sein
        '''
        # compute SPA assignment
        spa_estimate, iterations = snippets.decode_with_standart_log_bp(
            blockwise=False,
            rx=self.rx,
            EbN0=self.EbN0,
            code=self.code,
            bp=self.bp,
            max_iters=self.bp_iterations,
            rtol=self.rtol,
            atol=self.atol,
            infty=self.infty)
        self.assertEqual(np.sum(iterations == self.bp_iterations), 0, "BP did not converge for some cases") # assert that BP converged

        # compute Bitwise-MAP and compare
        map_estimate = snippets.bruteforce_bitwiseMAP_AWGNChannel(self.code, self.rx, self.EbN0)
        mpa_unequal_map = np.sum(np.logical_xor(spa_estimate, map_estimate), axis=1) > 0
        self.assertEqual(np.sum(mpa_unequal_map), 0, "some SPA estimates do not correspond to bitwise MAP")

    def test_bruteforce_MAP(self):
        map_correlation = snippets.bruteforce_blockwiseMAP_AWGNChannel(self.code, self.rx)
        log_factors = np.array([np.log(self.code.factors_AWGN(self.rx[idx], self.EbN0)) for idx in range(self.num_cws)])
        map_general = self.bp.bruteforce_MAP(log_factors)
        general_unequal_correlation = np.sum(np.logical_xor(map_general, map_correlation), axis=1) > 0
        self.assertEqual(np.sum(general_unequal_correlation), 0, "general MAP and correlation MAP do not correspond")