import unittest
import snippets
import numpy as np

class TestSymbolDetectionIsingModel(unittest.TestCase):

    num_cws = int(1e3)
    N = 4
    h = snippets.h_impulse_ising_model
    L = len(h) - 1
    
    def test_bruteforce_MAP_symbol_detection(self):
        '''
        tests snippets.bruteforce_MAP_symbol_detection by simulating noise free transmission and verifying that MAP corresponds to transmitted word
        '''
        tx_codewords = np.random.randint(2, size=(self.num_cws, self.N))
        tx = (-1)**tx_codewords
        y = np.row_stack([np.convolve(tx[cw_idx], self.h) for cw_idx in range(self.num_cws)]) # shape (num_cws, N + L)
        map_codewords = snippets.bruteforce_MAP_symbol_detection(y, self.h)
        map_codewords_unequal_tx_codewords = np.sum(np.logical_xor(tx_codewords, map_codewords), axis=1) > 0
        self.assertEqual(np.sum(map_codewords_unequal_tx_codewords), 0)