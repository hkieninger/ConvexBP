import numpy as np
import BlockCode
import BeliefPropagation

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

def simulateAWGNChannelTransmission(code : BlockCode.BlockCode, EbN0 : float, num_cws : int) -> np.ndarray:
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

def bruteforce_blockwiseMAP_AWGNChannel(code : BlockCode.BlockCode, rx : np.ndarray) -> np.ndarray:
    '''
    bruteforces the blockwise MAP solution to @rx { shape (num_cws, code.n) } if @code was used for transmission
    for AWGN channel: MAP codeword = codeword with maximum correlation (see NT1 Zusammenfassung, Seite 12)
    @return the most likely transmitted codewords { shape (num_cws, code.n) }
    '''
    codewords = code.codewords()
    bpsk_cws = (-1)**codewords
    # compute Blockwise-MAP estimate
    correlations = rx @ bpsk_cws.T # shape (num_cws, 2**k)
    map_estimate = codewords[np.argmax(correlations, axis=1)] # shape (num_cws, n)
    return map_estimate

def bruteforce_bitwiseMAP_AWGNChannel(code : BlockCode.BlockCode, rx : np.ndarray, EbN0 : float) -> np.ndarray:
    '''
    bruteforces the bitwise MAP solution to @rx { shape (num_cws, code.n) } if @code was used for transmission
    @return the most likely transmitted bits { shape (num_cws, code.n) }

    How the bitwise MAP is computed:
    argmax_xi P(xi | y) = argmax_xi sum_x P(xi, x | y) = argmax_xi sum_x P(y | xi, x) * P(xi | x) * P(x)
    = argmax_xi sum_x P(y | x) * 1{xi=(x)_i}

    P(xi = 1 | y) > P(xi = 0 | y)
    '''
    # compute Bitwise-MAP estimate
    codewords = code.codewords()
    bpsk_cws = (-1)**codewords

    EsN0_lin =  code.r * 10**(EbN0/10)
    sigma = 1 / np.sqrt(2 * EsN0_lin)

    y_minus_x_norm_squared = np.sum((rx[np.newaxis,:,:] - bpsk_cws[:,np.newaxis,:])**2, axis=2).T # shape (num_cws, 2**k)
    p_y_given_x = np.exp(-y_minus_x_norm_squared / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)) # shape (num_cws, 2**k)
    p_xi1_given_y = p_y_given_x @ codewords # proportional to p(xi = 1 | y) assuming uniform prior, shape (num_cws, n)
    p_xi0_given_y = p_y_given_x @ np.logical_not(codewords) # proportional to p(xi = 1 | y) assuming uniform prior, shape (num_cws, n)
    map_estimate = p_xi1_given_y > p_xi0_given_y # shape (num_cws, n)
    return map_estimate

def decode_with_standart_bp(blockwise : bool, rx : np.ndarray, EbN0 : float, 
                            code : BlockCode.BlockCode, bp : BeliefPropagation.BeliefPropagation, 
                            max_iters : int, convergence_threshold : float) -> tuple[np.ndarray, np.ndarray]:
    '''
    decodes @rx using standard BP (Bethe Approximation), if @blockwise = True MPA is used else SPA is used
    @return the decoded (code-)word and the amount of bp iterations until convergence, if iterations[idx] == max_iters BP didn't converge
    '''
    var_beliefs = np.empty((*rx.shape, 2))
    iterations = np.empty(var_beliefs.shape[0])
    gamma = bp.gammaBethe()
    for cw_idx in range(var_beliefs.shape[0]):
        (var_beliefs[cw_idx,:], _, _, iterations[cw_idx]) = bp.run_belief_propagation(
            max_iters=max_iters,
            convergence_threshold=convergence_threshold,
            factors=code.factors_AWGN(rx[cw_idx], EbN0),
            max_product=blockwise,
            gamma=gamma,
        )
    mpa_estimate = np.argmax(var_beliefs, axis=2) # decode with beliefs
    return (mpa_estimate, iterations)