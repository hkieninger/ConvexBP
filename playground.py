import snippets
import numpy as np
import matplotlib.pyplot as plt
import BeliefPropagation

with np.errstate(all="raise"):
    np.random.seed(45789)
    #np.random.seed(436789)

    EbN0 = 2
    max_iters = 10
    code = snippets.n5k2_acyclic

    y = snippets.simulateAWGNChannelTransmission(code, EbN0, 1)
    print(y)
    factors = code.factors_AWGN(y, EbN0)

    bp = BeliefPropagation.BeliefPropagation(code.adjacency_matrix(), 2)
    message_generator = bp.belief_propagation(factors, max_product=True, gamma=bp.gammaBethe())

    epsilons = np.empty(max_iters)
    (prev_v2f, prev_f2v) = np.copy(next(message_generator))
    iter = 1
    while iter < max_iters:
        (v2f, f2v) = np.copy(next(message_generator))
        epsilons[iter] = max(
            np.max(np.abs(f2v[bp.v_mask] - prev_f2v[bp.v_mask])), 
            np.max(np.abs(v2f[bp.v_mask] - prev_v2f[bp.v_mask]))
            )
        (prev_v2f, prev_f2v) = (v2f, f2v)
        iter += 1
    
    plt.plot(epsilons)
    plt.show()

    (vbelief, fbelief) = bp.messages2beliefs(v2f, f2v, factors, max_normalization=True)
    mpa_assignment = np.argmax(vbelief, axis=1)
    print(vbelief)

    map_assignment = np.squeeze(snippets.bruteforce_blockwiseMAP_AWGNChannel(code, y))

    print(f"MPA assignment: {mpa_assignment}, parity check: {code.satisfy_parity(mpa_assignment)}")
    print(f"MAP assignment: {map_assignment}, parity check: {code.satisfy_parity(map_assignment)}")


    # np.random.seed(456789)
    # #np.random.seed(436789)

    # code = snippets.n5k2_acyclic
    # num_cws = int(1e3)
    # EbN0 = 2
    # bp_iterations = 20
    # convergence_threshold = 1e-6

    # bp = BeliefPropagation.BeliefPropagation(adjacency_matrix=code.adjacency_matrix(), state_domain_size=2)
    # rx = snippets.simulateAWGNChannelTransmission(code, EbN0, num_cws)
    
    # # compute MPA assignment
    # var_beliefs = np.empty((num_cws, bp.n, 2))
    # gamma = bp.gammaBethe()
    # for cw_idx in range(num_cws):
    #     (var_beliefs[cw_idx,:], _, _, iters) = bp.run_belief_propagation(
    #         max_iters=bp_iterations,
    #         convergence_threshold=convergence_threshold,
    #         factors=code.factors_AWGN(rx[cw_idx], EbN0),
    #         max_product=True,
    #         gamma=gamma,
    #     )
    #     assert(iters < bp_iterations)
    # mpa_estimate = np.argmax(var_beliefs, axis=2) # decode with beliefs

    # # compute Blockwise-MAP and compare
    # map_estimate = snippets.bruteforce_blockwiseMAP_AWGNChannel(code, rx)
    # mpa_unequal_map = np.sum(np.logical_xor(mpa_estimate, map_estimate), axis=1) > 0
    # assert(np.sum(mpa_unequal_map) == 0)


