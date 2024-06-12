import test_BeliefPropagation
import BinaryBP

class TestBinaryBP(test_BeliefPropagation.TestBeliefPropagation):

    def setUp(self):
        super().setUp()
        self.bp = BinaryBP.BinaryBP(adjacency_matrix=self.code.adjacency_matrix())