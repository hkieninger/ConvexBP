import numpy as np

def binaryArray(stop : int, bit_width : int) -> np.ndarray:
    '''
    creates an array of binary numbers from 0 to stop-1
    returns a matrix of shape (stop, bit_width)
    [
    lsb_0, ..., msb_0;
        ...
    lsb_stop-1, ..., msb_stop-1;
    ]
    '''
    return ((np.arange(stop)[:, None] & (1 << np.arange(bit_width))) > 0).astype(int)