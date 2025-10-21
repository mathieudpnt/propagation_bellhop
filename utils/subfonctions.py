import numpy as np


def readline_1(fid, nb):
    return float(fid.readline().split()[nb])

def zeros(size, flag):
    if flag == 1 : # if complexe
        return np.zeros(size) + 1j * np.zeros(size)
    # if real
    return np.zeros( size )

def date_to_number(m,d):
    return (m - 1) * 30 + d
