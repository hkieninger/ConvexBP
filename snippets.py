import numpy as np
import BlockCode

'''
provides commonly needed snippets
- common block codes
- functions for simulation
'''

# simple acyclic code
H_n5k2_acyclic = np.array([[1, 0, 1, 0, 0],
                           [0, 1, 1, 0, 1],
                           [0, 0, 0, 1, 1]], dtype=int)
n5k2_acyclic = BlockCode.BlockCode(H_n5k2_acyclic)

# acyclic code
H_n8k8_acyclic = np.load('blockcodes/random_acyclic_LDPC.npy')
n8k8_acyclic = BlockCode.BlockCode(H_n8k8_acyclic)

# Hamming Code: contains cycles
H_hamming = np.array([[1, 0, 1, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0, 1, 1],
                      [0, 0, 0, 1, 1, 1, 1]], dtype=int)
n7k4_hamming = BlockCode.BlockCode(H_hamming)

def simulateAWGNChannelTransmission(code, EbN0, num_cws):
    '''
    simulates the transmission of @num_cws of @code over an AWGN channel with @EbN0
    @return a numpy array of shape (num_cws, code.n)
    '''
    # Generate random input bits.
    info_bits = np.random.randint(2, size=(num_cws, code.k))
    # Encoder
    code_bits = (info_bits @ code.G) % 2
    # BPSK mapping.
    tx = (-1) ** code_bits
    # Apply AWGN Channel.
    EsN0_lin =  code.r * 10**(EbN0/10)
    sigma = 1 / np.sqrt(2 * EsN0_lin)
    rx = tx + np.random.randn(*tx.shape) * sigma # shape (num_cws, n)
    return rx

def bruteforce_blockwiseMAP_AWGNChannel(code : BlockCode.BlockCode, rx):
    '''
    bruteforces the MAP solution to @rx { shape (num_cws, code.n) } if @code was used for transmission
    for AWGN channel: MAP codeword = codeword with maximum correlation (see NT1 Zusammenfassung, Seite 12)
    @return the most likely transmitted codewords { shape (num_cws, n) }
    '''
    codewords = code.codewords()
    bpsk_cws = (-1)**codewords
    # compute Blockwise-MAP estimate
    correlations = rx @ bpsk_cws.T # shape (num_cws, 2**k)
    map_estimate = codewords[np.argmax(correlations, axis=1)] # shape (num_cws, n)
    return map_estimate

def bruteforce_bitwiseMAP_AWGNChannel(code : BlockCode.BlockCode, rx):
    pass