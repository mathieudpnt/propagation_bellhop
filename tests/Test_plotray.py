from pathlib import Path

import numpy as np
import pytest

from utils import utils_acoustic_toolbox

def test_file_ray (tmp_path:Path) -> None:
    if tmp_path.exists():

    else :
        pytest.raises(FileNotFoundError)
    assert filename.exists()
    assert filename.suffix == ".ray"
    assert filename.stat().st_size > 0
    try: #voir si fichier corrompu
        _ = filename.read_text()
    except Exception as e:
        assert False


def test_text (filename : Path): #ok
    npoints=0
    nalpha=utils_acoustic_toolbox.plotray(filename)
    rmax=3021 #utils_acoustic_toolbox.write_env_file()
    fid= open(filename,'r')

    #header
    next(fid), next(fid), next(fid)
    next(fid)
    z_min=float(fid.readline())
    z_max=float(fid.readline())
    assert z_min<z_max
    next(fid)

    for _ibeam in range(nalpha):
        npoints+=1
        size_l = len( fid.readline() ) # departure angle of the  beam
        if size_l > 0: # loop until the end of the document
           nsteps =int(fid.readline().split()[0])
           assert nsteps != 0
           r = np.zeros(nsteps) # initiation of range array

           for nj in range(nsteps):
               r[nj] = float(fid.readline().split()[0])
               assert 0<=round(r[nj],0)<=rmax
        assert len( r ) == nsteps
        assert round(r[-1],0)==rmax
    assert npoints == nalpha