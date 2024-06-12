import numpy as np
from collections.abc import Iterator

class BeliefPropagation:
    '''
    General implementation of Belief Propagation using messages in linear domain as described in Weiss 2007
    -variables can have arbitrary state domain size
    -factors can be arbitrary non-negative functions
    '''

    def __init__(self, adjacency_matrix : np.ndarray, state_domain_size : int) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.s = state_domain_size
        (self.m, self.n) = adjacency_matrix.shape
        self.dv = np.sum(adjacency_matrix, axis=0)
        self.dv_max = np.max(self.dv)
        self.df = np.sum(adjacency_matrix, axis=1)
        self.df_max = np.max(self.df)

        # masks and mapping for belief propagation
        (self.row, self.col, self.f_mask, self.f2v_reshape, self.v_mask, self.v2f_reshape) = self.masks()

        # rows of self.message_shapes are used to reshape the messages in the factor node update
        self.message_shapes = np.ones((self.df_max, self.df_max + 1), dtype=int)
        self.message_shapes[:,0] = self.m
        self.message_shapes[:,1:] += np.eye(self.df_max, dtype=int) * (self.s - 1)
        pass

    def masks(self):
        """
        For the SP algorithm, we use 2 different arrangements of the messages:
        -Check node view (cview): m rows, each containing up to df_max messages from variable nodes
        -Variable node view (vview): n rows, each containing up to dv_max messages from check nodes
        
        This function calculates masks (in order to apply a computation only to valid entries) 
        and mappers (to switch between the 2 views).
        """
        (row, col) = np.nonzero(self.adjacency_matrix)
        row_cview = row
        col_cview = np.concatenate([np.arange(i) for i in np.diff(np.flatnonzero(np.r_[True,row[:-1]!=row[1:],True]))])
        f_mask = np.zeros((self.m, self.df_max), dtype=bool)
        f_mask[row_cview, col_cview] = True # Mask, which contains the valid entries of the check node view.

        f2v_reshape = np.argsort(col, kind='mergesort') # col_cview
        v2f_reshape = np.argsort(f2v_reshape, kind='mergesort')

        row_vview = np.sort(col)
        col_vview = np.concatenate([np.arange(i) for i in np.diff(np.flatnonzero(np.r_[True,row_vview[:-1]!=row_vview[1:],True]))])
        v_mask = np.zeros((self.n, self.dv_max), dtype=bool)
        v_mask[row_vview, col_vview] = True # Mask, which contains the valid entries of the variable node view.
        return row, col, f_mask, f2v_reshape, v_mask, v2f_reshape
    
    def c_var_Bethe(self):
        return 1 - self.dv
    
    def c_var_TrivialCBP(self):
        return np.zeros(self.n)
    
    def c_var_DefaultCBP(self):
        inverse_df = 1 / self.df
        c = np.zeros(self.n)
        for (factor, variable) in zip(self.row,self.col):
            c[variable] -= inverse_df[factor]
        return c 
    
    def gammaBethe(self):
        return np.ones(self.n)

    def gammaTrivialCBP(self):
        return self.dv
    
    def gammaDefaultCBP(self):
        return self.dv / (1 - self.c_var_DefaultCBP())


    def belief_propagation(self, factors : np.ndarray, max_product : bool, gamma : np.ndarray, temperature : float = 1, damping : float = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        '''
        creates a generator which yields the BP messages
        @factors: (n, s, s, ..., s) matrix, for factors with less than df_max arguments repeat the function along the unused axes
                     [-- df_max --]
        @max_product: if true MPA is used ele SPA
        @gamma: ConvexBP paramters computed from double counting numbers c_i of Free Energy Approximation (see Weiss 2007)
        @temperature: temperature of Free Energy Approximation (see Weiss 2007)
        @damping: damping used for message passing update, 0 no damping, 1 new message = previous message
        @return a generator
            generator yields (variable to factor, factor to variable)
            both message arrays are from variable perspective: shape (self.m, self.dv_max, self.s)
            messages are normalized such that maximum entry is 1
            to access the messages one can use self.v_mask
        '''
        if max_product:
            sum_operator = lambda array, axis: np.max(array, axis=axis, keepdims=True)
        else:
            sum_operator = lambda array, axis: np.sum(array, axis=axis, keepdims=True)
        
        v2f = np.ones((self.n, self.dv_max, self.s))
        f2v = np.ones((self.n, self.dv_max, self.s))

        v2f_fview = np.ones((self.m, self.df_max, self.s))
        f2v0_fview = np.empty((self.m, self.df_max, self.s))
        f2v0_vview = np.ones((self.n, self.dv_max, self.s))
        v2f0_vview = np.empty((self.n, self.dv_max, self.s))

        temperated_factors = factors**(1/temperature)
        shaped_gamma = gamma[:,np.newaxis,np.newaxis]

        while True:
            yield (v2f, f2v)
            # compute f2v0
            v2f_fview[self.f_mask] = v2f[self.v_mask,:][self.v2f_reshape,:]
            for extrinsic_port in range(self.df_max):
                accumulator = temperated_factors
                for port in range(self.df_max):
                    if port == extrinsic_port:
                        continue
                    accumulator = sum_operator(accumulator * v2f_fview[:,port,:].reshape(self.message_shapes[port,:]), port+1)
                f2v0_fview[:,extrinsic_port,:] = np.squeeze(accumulator)
            f2v0_vview[self.v_mask] = f2v0_fview[self.f_mask,:][self.f2v_reshape,:]

            # compute v2f0
            for extrinsic_port in range(self.dv_max):
                v2f0_vview[:,extrinsic_port,:] = np.prod(np.delete(f2v, extrinsic_port, axis=1), axis=1)
            v2f0_vview[np.logical_not(self.v_mask)] = 1

            # factor node update
            f2v = (f2v0_vview**shaped_gamma * v2f0_vview**(shaped_gamma-1))**(1-damping) * f2v**damping
            # variable node update
            v2f = (v2f0_vview**shaped_gamma * f2v0_vview**(shaped_gamma-1))**(1-damping) * v2f**damping

            # normalize messages
            f2v /= np.max(f2v, axis=2, keepdims=True)
            v2f /= np.max(v2f, axis=2, keepdims=True)

    def messages2beliefs(self, v2f : np.ndarray, f2v : np.ndarray, factors : np.ndarray, temperature : float = 1, max_normalization : bool = False) -> tuple[np.ndarray, np.ndarray]:
        '''
        @v2f: variable to factor messages obtained from belief_propagation
        @f2v: factor to variable messages
        @return: (variable_beliefs, factor_beliefs)
            variable_beliefs.shape = (self.n, self.s)
            factor_beliefs.shape = (self.m, self.s, self.s, ..., self.s)
                                            [------ self.df_max -------]
        '''
        if max_normalization:
            normalization_operator = lambda a: np.max(a, axis=1, keepdims=True)
        else:
            normalization_operator = lambda a: np.sum(a, axis=1, keepdims=True)

        # compute variable beliefs
        variable_beliefs = np.prod(f2v, axis=1)
        # normalize such that variable belief is a probability distribution
        variable_beliefs /= normalization_operator(variable_beliefs)
        
        # compute factor beliefs
        v2f_fview = np.ones((self.m, self.df_max, self.s))
        v2f_fview[self.f_mask] = v2f[self.v_mask,:][self.v2f_reshape,:]
        factor_beliefs = factors**(1/temperature)
        for port in range(self.df_max):
            factor_beliefs *= v2f_fview[:,port,:].reshape(self.message_shapes[port,:])
        # normalize such that factor belief is a probability distribution
        factor_beliefs /= normalization_operator(factor_beliefs.reshape((self.m, self.s**self.df_max))).reshape((self.m,) + self.df_max * (1,))
        
        return (variable_beliefs, factor_beliefs)

    def run_belief_propagation(self, max_iters, convergence_threshold, factors, max_product, gamma, temperature=1, damping=0):
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
        message_generator = self.belief_propagation(
            factors=factors,
            max_product=max_product,
            gamma=gamma,
            temperature=temperature,
            damping=damping
        )

        epsilons = np.zeros(max_iters)
        (prev_v2f, prev_f2v) = np.copy(next(message_generator))
        iter = 0
        while iter < max_iters:
            (v2f, f2v) = np.copy(next(message_generator))
            epsilons[iter] = max(
                np.max(np.abs(f2v[self.v_mask] - prev_f2v[self.v_mask])), 
                np.max(np.abs(v2f[self.v_mask] - prev_v2f[self.v_mask]))
                )
            (prev_v2f, prev_f2v) = (v2f, f2v)
            if epsilons[iter] < convergence_threshold:
                break
            iter += 1

        (variable_beliefs, factor_beliefs) = self.messages2beliefs(
            v2f=v2f, f2v=f2v, factors=factors, 
            temperature=temperature, 
            max_normalization=max_product)
        
        return (variable_beliefs, factor_beliefs, epsilons, iter)
    
    def satisfyAdmissibility(self, variable_beliefs, factor_beliefs, c_var, factors, temperature : float = 1, rtol : float = 1e-5, atol=1e-8) -> bool:
        '''
        only use for factor graphs with small self.n, complexity: O(self.s^self.n)
        checks whether beliefs satisfy admissibility condition (see Weiss 2007)
        '''
        probability_distribution_factor = np.ones((self.s,) * self.n)
        probability_distribution_marginals = np.ones((self.s,) * self.n)
        for factor in range(self.m):
            variables = np.nonzero(self.adjacency_matrix[factor,:])[0]
            shape = np.ones(self.n, dtype=int)
            shape[variables] = self.s
            idx_non_const_dims = (slice(None),) * self.df[factor] + (self.df_max - self.df[factor]) * (1,)
            probability_distribution_factor *= factors[factor][idx_non_const_dims].reshape(shape)
            probability_distribution_marginals *= factor_beliefs[factor][idx_non_const_dims].reshape(shape)
            # normalize to avoid numerical underflow
            probability_distribution_factor /= np.max(probability_distribution_factor)
            probability_distribution_marginals /= np.max(probability_distribution_marginals)
        
        variable_beliefs[variable_beliefs==0] = np.min(variable_beliefs[variable_beliefs != 0]) # TEST: remove later
        for variable in range(self.n):
            shape = np.ones(self.n, dtype=int)
            shape[variable] = self.s
            probability_distribution_marginals *= (variable_beliefs[variable]**c_var[variable]).reshape(shape)
            # normalize to avoid numerical underflow
            probability_distribution_marginals /= np.max(probability_distribution_marginals)
        
        probability_distribution_factor **= 1/temperature

        diff = np.abs(probability_distribution_factor - probability_distribution_marginals)
        tolerance = np.maximum(rtol * np.maximum(np.abs(probability_distribution_factor), np.abs(probability_distribution_marginals)), atol) # implementation as in math.isclose()
        return np.all(diff <= tolerance)

    def satisfyMarginalization(self, variable_beliefs, factor_beliefs, max=False, rtol : float = 1e-5, atol : float = 1e-8) -> bool:
        '''
        checks whether beliefs satisfy marginalization conditions (see Weiss 2007)
        '''
        if max:
            marginalisation_operator = lambda a, axis: np.max(a, axis=axis)
        else:
            marginalisation_operator = lambda a, axis: np.sum(a, axis=axis)

        for factor in range(self.m):
            variables = np.nonzero(self.adjacency_matrix[factor,:])[0]
            for var_idx in range(self.df[factor]):
                axes = list(range(self.df_max))
                del axes[var_idx]
                marginal = marginalisation_operator(factor_beliefs[factor], tuple(axes))
                marginal /= np.max(marginal)

                variable_belief = variable_beliefs[variables[var_idx]]
                variable_belief /= np.max(variable_belief)

                if np.any(np.abs(marginal - variable_belief) > np.maximum(rtol * np.maximum(np.abs(marginal), np.abs(variable_belief)), atol)):
                    return False
                
        return True
