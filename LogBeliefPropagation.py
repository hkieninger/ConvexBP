import numpy as np
import BeliefPropagation
from collections.abc import Iterator

class LogBeliefPropagation(BeliefPropagation.BeliefPropagation):
    '''
    General implementation of Belief Propagation using messages in log domain as described in Weiss 2007
    -variables can have arbitrary state domain size
    -factors can be arbitrary non-negative functions
    '''

    def __init__(self, adjacency_matrix : np.ndarray, state_domain_size : int) -> None:
        super().__init__(adjacency_matrix=adjacency_matrix, state_domain_size=state_domain_size)

    def log_belief_propagation(self, log_factors : np.ndarray, max_product : bool, gamma : np.ndarray, temperature : float = 1, damping : float = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        '''
        creates a generator which yields the BP messages
        @factors: (n, s, s, ..., s) matrix, for factors with less than df_max arguments repeat the function along the unused axes
                     [-- df_max --]
        @max_product: if true MPA is used ele SPA
        @gamma: ConvexBP paramters computed from double counting numbers c_i of Free Energy Approximation (see Weiss 2007)
        @temperature: temperature of Free Energy Approximation (see Weiss 2007)
        @damping: damping used for message passing update, 0 no damping, 1 new message = previous message
        @return a generator
            generator yields (variable to factor, factor to variable) messages in log domain
            both message arrays are from variable perspective: shape (self.m, self.dv_max, self.s)
            messages are normalized such that maximum entry is 0
            to access the messages one can use self.v_mask
        '''
        v2f = np.zeros((self.n, self.dv_max, self.s))
        f2v = np.zeros((self.n, self.dv_max, self.s))

        v2f_fview = np.zeros((self.m, self.df_max, self.s))
        f2v0_fview = np.empty((self.m, self.df_max, self.s))
        f2v0_vview = np.zeros((self.n, self.dv_max, self.s))
        v2f0_vview = np.empty((self.n, self.dv_max, self.s))

        shaped_gamma = gamma[:,np.newaxis,np.newaxis]

        while True:
            yield (v2f, f2v)
            # compute f2v0
            v2f_fview[self.f_mask] = v2f[self.v_mask,:][self.v2f_reshape,:]
            for extrinsic_port in range(self.df_max):
                accumulator = log_factors / temperature
                for port in range(self.df_max):
                    if port == extrinsic_port:
                        continue
                    if max_product:
                        accumulator = np.max(accumulator + v2f_fview[:,port,:].reshape(self.message_shapes[port,:]), axis=port+1, keepdims=True)
                    else:
                        accumulator += v2f_fview[:,port,:].reshape(self.message_shapes[port,:])
                if max_product:
                    f2v0_fview[:,extrinsic_port,:] = np.squeeze(accumulator)
                else:
                    axes = list(range(1,self.df_max+1))
                    del axes[extrinsic_port]
                    axes = tuple(axes)
                    f2v0_fview[:,extrinsic_port,:] = np.log(np.sum(np.exp(accumulator), axis=axes))
            f2v0_vview[self.v_mask] = f2v0_fview[self.f_mask,:][self.f2v_reshape,:]

            # compute v2f0
            for extrinsic_port in range(self.dv_max):
                v2f0_vview[:,extrinsic_port,:] = np.sum(np.delete(f2v, extrinsic_port, axis=1), axis=1)
            v2f0_vview[np.logical_not(self.v_mask)] = 0

            power_zero_mask = gamma - 1 == 0 # to handle 0 * -infty = 0
            # factor node update
            v2f0_vview_masked = np.copy(v2f0_vview)
            v2f0_vview_masked[power_zero_mask] = 0
            f2v = (f2v0_vview * shaped_gamma + v2f0_vview_masked * (shaped_gamma - 1)) * (1 - damping) + f2v * damping
            # variable node update
            f2v0_vview_masked = np.copy(f2v0_vview)
            f2v0_vview_masked[power_zero_mask] = 0
            v2f = (v2f0_vview * shaped_gamma + f2v0_vview_masked * (shaped_gamma - 1)) * (1 - damping) + v2f * damping

            # normalize messages such that maximum entry is 0
            f2v -= np.max(f2v, axis=2, keepdims=True)
            v2f -= np.max(v2f, axis=2, keepdims=True)

    def log_messages_2_log_beliefs(self, v2f : np.ndarray, f2v : np.ndarray, log_factors : np.ndarray, temperature : float = 1) -> tuple[np.ndarray, np.ndarray]:
        '''
        @v2f: variable to factor messages obtained from belief_propagation
        @f2v: factor to variable messages
        @return: (variable_beliefs, factor_beliefs)
            variable_beliefs.shape = (self.n, self.s)
            factor_beliefs.shape = (self.m, self.s, self.s, ..., self.s)
                                            [------ self.df_max -------]
        '''
        # compute variable beliefs
        variable_beliefs = np.sum(f2v, axis=1)
        # normalize such that maximum entry is 0
        variable_beliefs -= np.max(variable_beliefs, axis=1, keepdims=True)
        
        # compute factor beliefs
        v2f_fview = np.zeros((self.m, self.df_max, self.s))
        v2f_fview[self.f_mask] = v2f[self.v_mask,:][self.v2f_reshape,:]
        factor_beliefs = log_factors / temperature
        for port in range(self.df_max):
            factor_beliefs += v2f_fview[:,port,:].reshape(self.message_shapes[port,:])
        # normalize such that maximum entry of each factor is 0
        factor_beliefs -= np.max(factor_beliefs.reshape((self.m, self.s**self.df_max)), axis=1, keepdims=True).reshape((self.m,) + self.df_max * (1,))
        
        return (variable_beliefs, factor_beliefs)

    def run_log_belief_propagation(self, max_iters, rtol, atol, infty, log_factors, max_product, gamma, temperature=1, damping=0):
        '''
        runs belief propagation until convergence or @max_iters has been reached
        @convergence_threshold maximum difference between previous and current messages for them to be considered identical
        @other_arguments see belief_propagation
        @return (variable beliefs, factor beliefs, epsilons, iter)
            -variable beliefs: shape (self.n, self.s)
            -factor beliefs: shape (self.m, self.s, self.s, ..., self.s), function repeated along unused dimensions
                                            [------ self.df_max -------]
            -epsilons: maximum difference between previous and current messages for the iterations, shape (self.max_iters,)
            -iter: amount of iterations until convergence, if iter == max_iters BP didn't converge
        '''
        message_generator = self.log_belief_propagation(
            log_factors=log_factors,
            max_product=max_product,
            gamma=gamma,
            temperature=temperature,
            damping=damping
        )

        (prev_v2f, prev_f2v) = np.copy(next(message_generator))
        iter = 0
        while iter < max_iters:
            (v2f, f2v) = np.copy(next(message_generator))
            infmask_v2f = prev_v2f[self.v_mask] <= -infty
            infmask_f2v = prev_f2v[self.v_mask] <= -infty
            finmask_v2f = np.logical_not(infmask_v2f)
            finmask_f2v = np.logical_not(infmask_f2v)
            # check for convergence of fix-points in -infty in log domain (0 in linear domain)
            inf_fixed_points_converged = \
                np.all(v2f[self.v_mask][infmask_v2f] <= prev_v2f[self.v_mask][infmask_v2f]) and \
                np.all(f2v[self.v_mask][infmask_f2v] <= prev_f2v[self.v_mask][infmask_f2v])
            # check for convergence of finite fix-points
            fin_fixed_points_converged = \
                np.allclose(v2f[self.v_mask][finmask_v2f], prev_v2f[self.v_mask][finmask_v2f], rtol, atol) and \
                np.allclose(f2v[self.v_mask][finmask_f2v], prev_f2v[self.v_mask][finmask_f2v], rtol, atol)
            # stop if all messages converged
            if inf_fixed_points_converged and fin_fixed_points_converged:
                break

            (prev_v2f, prev_f2v) = (v2f, f2v)
            iter += 1

        (variable_beliefs, factor_beliefs) = self.log_messages_2_log_beliefs(
            v2f=v2f, f2v=f2v, log_factors=log_factors, 
            temperature=temperature)
        
        return (variable_beliefs, factor_beliefs, iter)
    
    def bruteforce_MAP(self, log_factors):
        log_distribution = np.zeros((log_factors.shape[0],) + self.n * (self.s,))
        for factor in range(self.m):
            variables = np.nonzero(self.adjacency_matrix[factor,:])[0]
            shape = np.ones(1 + self.n, dtype=int)
            shape[0] = log_factors.shape[0]
            shape[1 + variables] = self.s
            idx_non_const_dims = (slice(None),) * (1 + self.df[factor]) + (self.df_max - self.df[factor]) * (1,)
            log_distribution += log_factors[:,factor][idx_non_const_dims].reshape(shape)
        argmax_flattened = np.argmax(log_distribution.reshape(log_factors.shape[0],self.s**self.n), axis=1)
        argmax = np.zeros((log_factors.shape[0],self.n), dtype=int)
        for dim in range(self.n):
            argmax[:,self.n - 1 - dim] = argmax_flattened % self.s
            argmax_flattened //= self.s
        return argmax