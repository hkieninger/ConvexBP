import snippets
import numpy as np
import matplotlib.pyplot as plt
import BeliefPropagation

with np.errstate(all="raise"):
    np.random.seed(456789)

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

    (vbelief, fbelief) = bp.messages2beliefs(v2f, f2v, factors)
    mpa_assignment = np.argmax(vbelief, axis=1)
    print(vbelief)

    map_assignment = snippets.bruteforce_blockwiseMAP_AWGNChannel(code, y)

    print(mpa_assignment)
    print(map_assignment)
    print(y)




