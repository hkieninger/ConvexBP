import numpy as np
from collections.abc import Iterator
import BeliefPropagation

class BinaryBP(BeliefPropagation.BeliefPropagation):
    '''
    implementation of BP for the case of binary variables
    for binary variables we can use LLR to efficiently represent a message by a single value
    '''

    def __init__(self, adjacency_matrix: np.ndarray) -> None:
        super().__init__(adjacency_matrix, state_domain_size=2)

    def llr_belief_propagation(self, log_factors, max_product, gamma, temperature=1, damping=0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        '''
        belief propagation with LLRs
        '''
        v2f = np.zeros((self.n, self.dv_max))
        f2v = np.zeros((self.n, self.dv_max))

        v2f_fview = np.zeros((self.m, self.df_max))
        f2v0_fview = np.empty((self.m, self.df_max))
        f2v0_vview = np.zeros((self.n, self.dv_max))
        v2f0_vview = np.empty((self.n, self.dv_max))

        temperated_factors = log_factors / temperature
        shaped_gamma = gamma[:,np.newaxis]

        while True:
            yield (v2f, f2v)
            # compute f2v0
            v2f_fview[self.f_mask] = v2f[self.v_mask][self.v2f_reshape]
            diverging_lr = np.isposinf(v2f_fview)
            v2f_fview[diverging_lr] = 0
            for extrinsic_port in range(self.df_max):
                accumulator = np.copy(temperated_factors)
                for port in range(self.df_max):
                    if port == extrinsic_port:
                        continue
                    accumulator[(slice(None),) * (port + 1) + (0,) + (slice(None),) * (self.df_max - port - 1)] += \
                        v2f_fview[:,port].reshape((self.m,) + (1,) * (self.df_max - 1))
                    accumulator[diverging_lr[:,port]][(slice(None),) * (port + 1) + (1,) + (slice(None),) * (self.df_max - port - 1)] += -float('inf')

                axes = list(range(1,self.df_max+1))
                del axes[extrinsic_port]
                axes = tuple(axes)
                if max_product:
                    messages_log = np.max(accumulator, axis=axes)
                else:
                    messages_log = np.log(np.sum(np.exp(accumulator), axis=axes))
                f2v0_fview[:,extrinsic_port] = messages_log[:,0] - messages_log[:,1]
            f2v0_vview[self.v_mask] = f2v0_fview[self.f_mask][self.f2v_reshape]

            # compute v2f0
            for extrinsic_port in range(self.dv_max):
                v2f0_vview[:,extrinsic_port] = np.sum(np.delete(f2v, extrinsic_port, axis=1), axis=1)
            v2f0_vview[np.logical_not(self.v_mask)] = 0

            # factor node update
            f2v = (f2v0_vview * shaped_gamma + v2f0_vview * (shaped_gamma - 1)) * (1 - damping) + f2v * damping
            # variable node update
            v2f = (v2f0_vview * shaped_gamma + f2v0_vview * (shaped_gamma - 1)) * (1 - damping) + v2f * damping
    
    @staticmethod
    def llr2lin(llr, max_normalization=False):
        lin = np.empty((*llr.shape, 2))
        lr = np.exp(llr)
        diverging_lr = np.isposinf(lr)
        not_divering_lr = np.logical_not(diverging_lr)
        lin[:,:,1] = 1 / (1 + lr)
        lin[not_divering_lr,0] = lr[not_divering_lr] * lin[not_divering_lr,1]
        lin[diverging_lr,0] = 1
        if max_normalization:
            lin /= np.max(lin,axis=2,keepdims=True)
        return lin

    def belief_propagation(self, factors: np.ndarray, max_product: bool, gamma: np.ndarray, temperature: float = 1, damping: float = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        llr_message_generator = self.llr_belief_propagation(np.log(factors), max_product, gamma, temperature, damping)
        while True:
            (v2f_llr, f2v_llr) = next(llr_message_generator)
            yield (BinaryBP.llr2lin(v2f_llr, max_normalization=True), BinaryBP.llr2lin(f2v_llr, max_normalization=True))

    
