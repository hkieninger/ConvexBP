from numpy import ndarray
import BeliefPropagation

class BinaryBP(BeliefPropagation.BeliefPropagation):
    '''
    implementation of BP for the case of binary variables
    for binary variables we can use LLR to efficiently represent a message by a single value
    '''

    def __init__(self, adjacency_matrix: ndarray) -> None:
        super().__init__(adjacency_matrix)

    def belief_propagation(self, factors, max_product, gamma, temperature=1, damping=0) -> BeliefPropagation.Iterator[tuple[ndarray, ndarray]]:
        return super().belief_propagation(factors, max_product, gamma, temperature, damping)
    
