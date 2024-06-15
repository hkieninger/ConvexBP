import numpy as np
import utils
import BlockCode
import BeliefPropagation

'''
provides commonly needed snippets
- common block codes
- functions for simulation
- Ising Model representing linear interference channel
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

H_24_12_golay = golay_24_12 = (np.load('blockcodes/opt_golay24_12_with_ends_even_more_weights.npz')['Hopt']).astype(int)
n24_k12_golay = BlockCode.BlockCode(H_24_12_golay)


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

def decode_with_standart_log_bp(blockwise : bool, rx : np.ndarray, EbN0 : float, 
                            code : BlockCode.BlockCode, bp : BeliefPropagation.BeliefPropagation, 
                            max_iters : int, rtol : float, atol : float, infty : float) -> tuple[np.ndarray, np.ndarray]:
    '''
    decodes @rx using standard BP (Bethe Approximation), if @blockwise = True MPA is used else SPA is used
    @return the decoded (code-)word and the amount of bp iterations until convergence, if iterations[idx] == max_iters BP didn't converge
    '''
    var_beliefs = np.empty((*rx.shape, 2))
    iterations = np.empty(var_beliefs.shape[0])
    gamma = bp.gammaBethe()
    for cw_idx in range(var_beliefs.shape[0]):
        (var_beliefs[cw_idx,:], _, iterations[cw_idx]) = bp.run_log_belief_propagation(
            max_iters=max_iters,
            rtol=rtol,
            atol=atol,
            infty= infty,
            log_factors=np.log(code.factors_AWGN(rx[cw_idx], EbN0)),
            max_product=blockwise,
            gamma=gamma,
        )
    mpa_estimate = np.argmax(var_beliefs, axis=2) # decode with beliefs
    return (mpa_estimate, iterations)

# channel impulse response of symbol detection Ising model recommended by Luca
h_impulse_ising_model = np.array([0.408, 0.815, 0.408])

def symbol_detection_ising_model(y : np.ndarray, h : np.ndarray  = h_impulse_ising_model, sigma : float = np.sqrt(0.1)):
    '''
    computes the adjacency_matrix and log_factors associated with the reception of @y from an linear interference AWGN channel (see Schmid 2023, section 2.3)
    we assume a real impulse response @h and BPSK modulation
    @y words at receiver, shape (num_cws, N + L)
    @h impulse response of channel, shape (L + 1)
    @sigma noise variance of channel
    @return (adjacency_matrix, log_factors) 
    '''
    (num_cws, N_plus_L) = y.shape
    L = len(h) - 1 # len(h) = L + 1 (Definition as in Schmid 2023, section 2.3)
    N = N_plus_L - L # symbols per word
    # compute symbol detection Ising models
    H = np.column_stack([np.roll(np.concatenate((h, np.zeros(N - 1))), shift) for shift in range(N)]) # shape (N + L, N)
    G = H.T @ H
    x = (H.T @ y.T).T # shape (num_cws, N)
    # remove elements of diagonal and below diagonal
    G = np.triu(G,k=1)
    # compute adjacency matrix
    (rows, cols) = np.nonzero(G)
    degree_2_factors_adjacency = np.zeros((len(rows), N), dtype=int)
    for idx, (row,col) in enumerate(zip(rows,cols)):
        degree_2_factors_adjacency[idx,row] = 1
        degree_2_factors_adjacency[idx,col] = 1
    adjacency_matrix = np.concatenate((np.eye(N, dtype=int), degree_2_factors_adjacency), dtype=int)
    # compute the factors
    degree_1_factors = 2 * x.reshape(y.shape[0], N, 1, 1) * np.array([[1, 1], [-1, -1]]).reshape(1,1,2,2) / sigma**2 # shape (num_cws, N, 2, 2)
    degree_2_factors = -2 * G[G != 0].reshape(-1, 1, 1) * np.array([[1, -1], [-1, 1]]).reshape(1,2,2) / sigma**2 # shape (?, 2, 2)
    log_factors = np.array([np.concatenate((degree_1_factors[cw_idx], degree_2_factors), axis=0) for cw_idx in range(num_cws)])
    return adjacency_matrix, log_factors

def random_symbol_detection_ising_model(num_cws : int, N : int = 4, h : np.ndarray  = h_impulse_ising_model, sigma : float = np.sqrt(0.1)):
    '''
    simulates the transmission of @num_cws words of lenth @N over an linear interference channel @h, @sigma
    @return (y, adjacency_matrix, log_factors)
        -y: symbols at receiver affected by impulse response and noise
        -adjacency_matrix: adjacency_matrix of factor graph corresponding to the problem of MAP estimation (see Schmid 2023, section 2.3)
        -log_factors: array of array of factors correspondng to the problem of MAP estimation with a factor graph for each codeword
    '''
    L = len(h) - 1
    # simulate transmission of @tx over AWGN channel with @sigma
    tx = (-1)**np.random.randint(2, size=(num_cws, N))
    y = np.row_stack([np.convolve(tx[cw_idx], h) + np.random.randn(N + L) * sigma / np.sqrt(2) for cw_idx in range(num_cws)]) # shape (num_cws, N + L)
    return y, *symbol_detection_ising_model(y, h, sigma)

def bruteforce_MAP_symbol_detection(y : np.ndarray, h : np.ndarray = h_impulse_ising_model):
    '''
    bruteforces the MAP solution when @y is the observation at the reiver of an linear interference AWGN channel with impulse response @h
    @y shape (num_cws, N + L)
    @h shape (L + 1)
    TODO: implement Viterbi for more efficient MAP computation
    '''
    (num_cws, N_plus_L) = y.shape
    L = len(h) - 1
    N = N_plus_L - L
    codewords = utils.binaryArray(2**N, N) # shape (2**N, N)
    H = np.column_stack([np.roll(np.concatenate((h, np.zeros(N - 1))), shift) for shift in range(N)]) # shape (N + L, N)
    noise_free_channel_outputs = (H @ ((-1)**codewords).T) # shape (N + L, 2**N)
    diff = y.reshape(num_cws, N + L, 1) - noise_free_channel_outputs.reshape(1, N + L, 2**N) # shape (num_cws, N+L, 2**N)
    norm_squared = np.sum(diff**2, axis=1) # shape (num_cws, 2**N)
    return codewords[np.argmin(norm_squared, axis=1)]