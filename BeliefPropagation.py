import numpy as np
from collections.abc import Iterator

class BeliefPropagation:

    def __init__(self, adjacency_matrix : np.ndarray, state_domain_size : int) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.s = state_domain_size
        (self.m, self.n) = adjacency_matrix.shape
        self.dv = np.sum(adjacency_matrix, axis=0)
        self.dv_max = np.max(self.dv)
        self.df = np.sum(adjacency_matrix, axis=1)
        self.df_max = np.max(self.df)

        # 
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
    
    def gammaBethe(self):
        return np.ones(self.n)

    def gammaTrivialCBP(self):
        return self.dv
    
    def gammaDefaultCBP(self):
        inverse_df = 1 / self.df
        c = np.zeros(self.n)
        for (factor, variable) in zip(self.row,self.col):
            c[variable] -= inverse_df[factor]
        return self.dv / (1 - c)


    def belief_propagation(self, factors, max_product, gamma, temperature=1, damping=0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        '''
        @factors: (n, s, s, ..., s) matrix, for factors with less than df_max arguments, place the factor in the first df dimension and set the remaining entries to 0
                     [-- df_max --] 
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
            v2f = (f2v0_vview**shaped_gamma * v2f0_vview**(shaped_gamma-1))**(1-damping) * v2f**damping

            # normalize messages
            f2v /= np.max(f2v, axis=2)[:,:,np.newaxis]
            v2f /= np.max(v2f, axis=2)[:,:,np.newaxis]

    def messages2beliefs(self, v2f, f2v, factors, temperature=1):
        '''
        @return: (variable_beliefs, factor_beliefs)
            variable_beliefs.shape = (self.n, self.s)
            factor_beliefs.shape = (self.m, self.s, self.s, ..., self.s)
                                            [------ self.df_max -------]
        '''
        # compute variable beliefs
        variable_beliefs = np.prod(f2v, axis=1)
        # normalize such that variable belief is a probability distribution
        variable_beliefs /= np.sum(variable_beliefs, axis=1, keepdims=True)
        
        # compute factor beliefs
        v2f_fview = np.ones((self.m, self.df_max, self.s))
        v2f_fview[self.f_mask] = v2f[self.v_mask,:][self.v2f_reshape,:]
        factor_beliefs = factors**(1/temperature)
        for port in range(self.df_max):
            factor_beliefs *= v2f_fview[:,port,:].reshape(self.message_shapes[port,:])
        # normalize such that factor belief is a probability distribution
        factor_beliefs /= np.sum(factor_beliefs.reshape((self.m, self.s**self.df_max)), axis=1).reshape((self.m,) + self.df_max * (1,))
        
        return (variable_beliefs, factor_beliefs)

