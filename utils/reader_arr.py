
from pathlib import Path

from pandas.core.arrays.numeric import NumericArray
from pandas.core.internals.construction import nested_data_to_arrays

from utils.sub_read_env import (
    check_eq,
    read_angle,
    read_bot_prop,
    read_depth,
    read_env_param,
    read_md,
    read_prof,
    read_run_type,
    read_z,
)


def read_env(file: Path) -> list :
    """Check the arrival file created by bellhop.

    Parameters
    ----------
    file : Path
        Path to the .arr file.

    Returns
    -------
    content : list
        The contents of the .arr file.

    """
    if file.suffix != ".arr":
        msg = f"{file} is not a .arr file"
        raise ValueError(msg)

    if not file.exists():
        msg = f"{file} does not exist"
        raise FileNotFoundError(msg)

    content = file.read_text().splitlines()

    if not content:
        msg = f"{file} is empty"
        raise ValueError(msg)


    dimension = content[0]
    # 2D test
    frequency=float(content[1])

    nb_src, src_z =content[2].split()
    nb_rcv_z, rcv_z=content[3].split()
    nb_rcv_r, rcv_r=content[4].split()

    narr=int(content[5])
    narr=int(content[6])
    # nb lines = narr

    i=7
    amp=[]
    phase=[]
    delay_re=[]
    delay_im=[]
    src_ang=[]
    rcv_ang=[]
    nb_top_bnc=[]
    nb_bot_bnc = []
    a=i+narr
    while i < a:
        line = content[i].split()
        amp.append(float(line[0]))
        phase.append(float(line[1]))
        delay_re.append(float(line[2]))
        delay_im.append(float(line[3]))
        src_ang.append(float(line[4]))
        rcv_ang.append(float(line[5]))
        nb_top_bnc.append(int(line[6]))
        nb_bot_bnc.append(int(line[7]))
        i += 1

    # src_ang must be decreasing
    # check len each variable == narr
    #
    arr_data = {"amp" : amp,
                "phase": phase,
                "delay_re" : delay_re,
                "delay_im" : delay_im,
                "src_ang" : src_ang,
                "rcv_ang" : rcv_ang,
                "nb_top_bnc" : nb_top_bnc,
                "nb_bot_bnc" : nb_bot_bnc,
                }

    return content,arr_data