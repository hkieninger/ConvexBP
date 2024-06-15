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

def log(a):
    '''
    computes the log of @a
    considers log(0) as extra case to avoid triggering np.errstate
    '''
    log = np.empty_like(a)
    mask = (a == 0)
    not_mask = np.logical_not(mask)
    log[mask] = -float('inf')
    log[not_mask] = np.log(a[not_mask])
    return log
