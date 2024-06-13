import numpy as np
import galois
import utils

class BlockCode:

    def __init__(self, parity_check_matrix : np.ndarray) -> None:
        self.H = parity_check_matrix
        (self.m, self.n) = self.H.shape
        self.dv = np.sum(parity_check_matrix, axis=0)
        self.dv_max = np.max(self.dv)
        self.dc = np.sum(parity_check_matrix, axis=1)
        self.dc_max = np.max(self.dc)
        
        # Compute generator matrix G.
        self.G = self.make_gen()
        self.k = self.G.shape[0]
        self.r = self.k / self.n

    @classmethod
    def fromAlistFile(cls, alist_filename : str):
        return cls(BlockCode.read_alist_file(alist_filename))
    
    @staticmethod
    def read_alist_file(filename : str) -> np.ndarray:
        """
        This function reads in an alist file and creates the
        corresponding parity check matrix H. The format of alist
        files is described at:
        http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
        """
        myfile    = open(filename,'r')
        data      = myfile.readlines()
        size      = str.split(data[0])
        numCols   = int(size[0])
        numRows   = int(size[1])
        H = np.zeros((numRows,numCols), dtype=np.int8)
        for lineNumber in np.arange(4,4+numCols):
          line = np.fromstring(data[lineNumber], dtype=int, sep=' ')
          for index in line[line != 0]:
            H[int(index)-1,lineNumber-4] = 1
        return H
    
    def make_gen(self) -> np.ndarray:
        """
        This function computes the corresponding generator matrix G to the given
        parity check matrix H.
        """
        GF = galois.GF(2)
        G = np.array(GF(self.H).null_space(), dtype=np.int8)
        # Sanity check.
        assert not np.any((G @ self.H.T) % 2)
        return G
    
    def satisfy_parity(self, words : np.ndarray) -> np.ndarray:
        '''
        checks wether @words is/are a codeword
        '''
        return np.logical_not(np.any(((self.H @ words.T) % 2).T, axis=-1))
    
    def codewords(self):
       '''
       only use this method for small k
       @return all codewords of the BlockCode, shape (2**k, n)
       '''
       return (utils.binaryArray(2**self.k, self.k) @ self.G) % 2
    
    def adjacency_matrix(self) -> np.ndarray:
       '''
       @return the adjacency matrix of a factor graph corresponding to the block code
       in comparision to the parity-check matrix H, the adjacency matrix not only considers the checknodes but aswell the dongles as factors
       '''
       return np.concatenate((self.H, np.eye(self.n, dtype=int)), axis=0)
    
    def checknode_factors(self) -> np.ndarray:
        '''
        @return a numpy array representing the check node factors which can be used with the BeliefPropagation class
        '''
        factors = np.empty(shape=(self.m,) + self.dc_max * (2,))
        for factor_idx in range(self.m):
            arange = np.arange(2**self.dc[factor_idx], dtype=int)
            checks = np.ones_like(arange)
            for _ in range(self.dc[factor_idx]):
               checks += arange & 1
               arange >>= 1
               checks %= 2
            factor = checks.reshape(self.dc[factor_idx] * (2,))
            for _ in range(self.dc_max - self.dc[factor_idx]):
               factor = np.stack((factor,) * 2, axis=-1)
            factors[factor_idx] = factor
        return factors
            
    def dongle_factors_AWGN(self, y : np.ndarray, EbN0 : float) -> np.ndarray:
        '''
        @return a numpy array representing the dongle factors for the received codeword @y from an AWGN channel of @EbN0
        '''
        factors = np.empty((self.n, 2))

        EsN0_lin =  self.r * 10**(EbN0/10)
        sigma = 1 / np.sqrt(2 * EsN0_lin)
        f_y_x = lambda x: 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(y - (-1)**x)**2 / (2 * sigma**2))
        factors[:,0] = f_y_x(0)
        factors[:,1] = f_y_x(1)
        for _ in range(self.dc_max-1):
           factors = np.stack((factors,) * 2, axis=-1)
        return factors
    
    def factors_AWGN(self, y : np.ndarray, EbN0 : float) -> np.ndarray:
       '''
       @return a numpy array representing the checknode and dongle factors of the received codeword @y from an AWGN channel of @EbN0
       both checknodes and dongles are considered as factors
       '''
       return np.concatenate((self.checknode_factors(), self.dongle_factors_AWGN(y, EbN0)), axis=0)


