import snippets
import numpy as np
import matplotlib.pyplot as plt
import BinaryBP

with np.errstate(all="warn"):
    np.random.seed(45789)

    num_cws = int(1e5)
    EbN0 = 2
    max_iters = 20
    code = snippets.n5k2_acyclic

    rx = snippets.simulateAWGNChannelTransmission(code, EbN0, num_cws)
    rx = rx[np.array([1,259]),:]
    bp = BinaryBP.BinaryBP(code.adjacency_matrix())

    var_beliefs = np.empty((*rx.shape, 2))
    iterations = np.empty(var_beliefs.shape[0])
    gamma = bp.gammaBethe()
    for cw_idx in range(var_beliefs.shape[0]):
        (var_beliefs[cw_idx,:], _, _, iterations[cw_idx]) = bp.run_belief_propagation(
            max_iters=max_iters,
            convergence_threshold=1e-6,
            factors=code.factors_AWGN(rx[cw_idx], EbN0),
            max_product=True,
            gamma=gamma,
            damping=0
        )
    converged = iterations < max_iters
    print(np.nonzero(converged == False))
    mpa_assignment = np.argmax(var_beliefs, axis=2) # decode with beliefs

    map_assignment = snippets.bruteforce_blockwiseMAP_AWGNChannel(code, rx)

    # print(f"MPA assignment: {mpa_assignment}, parity check: {code.satisfy_parity(mpa_assignment)}")
    # print(f"MAP assignment: {map_assignment}, parity check: {code.satisfy_parity(map_assignment)}")


