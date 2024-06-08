from collections.abc import Iterator
import BinaryBP

class LDPCBP(BinaryBP.BinaryBP):

    def __init__(self, adjacency_matrix: BinaryBP.ndarray) -> None:
        super().__init__(adjacency_matrix)

    def belief_propagation(self, factors, max_product, gamma, temperature=1, damping=0) -> Iterator[tuple[ndarray, ndarray]]:
        return super().belief_propagation(factors, max_product, gamma, temperature, damping)