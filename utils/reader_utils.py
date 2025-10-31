"""Read files."""
import itertools
from pathlib import Path

import numpy as np
from core_utils import (
    check_elem_num,
    check_empty_file,
    check_file_exist,
    check_len_list,
    check_suffix,
)
from numpy import ndarray


def read_env(file: Path) -> dict:
    """Check the environmental file created by bellhop.

    Parameters
    ----------
    file : Path
        Path to the environmental file.

    Returns
    -------
    content : list
        The contents of the environmental file.

    """
    check_file_exist(file)
    check_suffix(file, ".env")
    check_empty_file(file)
    content = file.read_text(encoding="utf-8").splitlines()

    title = content[0].strip("'\"")
    frequency = int(content[1])

    number_media = int(content[2])
    check_media(number_media)

    env_opt = content[3].strip("'\"")

    env_param = read_env_param(content[4])

    z_min, z_max = read_depth(content[5])

    i = 6
    depth_profile, sound_speed_profile = [], []
    while read_z(content[i], z_min) < z_max:
        depth_profile.append(read_z(content[i], z_min))
        sound_speed_profile.append(read_soundspeed(content[i]))
        i += 1

    if read_z(content[i], z_min) != z_max:
        msg = "Wrong last depth value"
        raise ValueError(msg)
    depth_profile.append(read_z(content[i], z_min))
    sound_speed_profile.append(read_soundspeed(content[i]))

    check_soundspeed_profile(depth_profile)

    bottom_cond = content[i + 1]
    bottom_prop = read_bottom_properties(content[i + 2], z_max)

    nb_src = int(content[i + 3])
    src_z = float(content[i + 4].strip(" /"))
    nb_rcv_z = int(content[i + 5])

    if nb_rcv_z == 1:
        rcv_z = float(content[i + 6].strip(" /"))
    else:
        rcv_z = tuple(float(elem) for elem in content[i + 6].strip(" /").split())

    nb_rcv_r = int(content[i + 7])

    if nb_rcv_z == 1:
        rcv_r = float(content[i + 8].strip(" /"))
    else:
        rcv_r = tuple(float(elem) for elem in content[i + 8].strip(" /").split())

    run_type = content[i + 9]
    check_run_type(run_type)

    nb_beam = int(content[i + 10])

    angle_inf, angle_sup = read_angle(content[i + 11])

    info = content[i + 12]

    return {
        "title": title,
        "frequency": frequency,
        "number_media": number_media,
        "env_opt": env_opt,
        "env_param": env_param,
        "depth_profile": depth_profile,
        "sound_speed_profile": sound_speed_profile,
        "bottom_condition": bottom_cond,
        "bottom_property": bottom_prop,
        "nb_src": nb_src,
        "source_depth": src_z,
        "nb_rcv_z": nb_rcv_z,
        "rcv_z": rcv_z,
        "nb_rcv_r": nb_rcv_r,
        "rcv_r": rcv_r,
        "run_type": run_type,
        "nb_beam": nb_beam,
        "angles": (angle_inf, angle_sup),
        "info": info
    }


def read_arr(file: Path) -> list:
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
    check_file_exist(file)
    check_suffix(file, ".arr")
    check_empty_file(file)
    content = file.read_text(encoding="utf-8").splitlines()

    dimension = content[0].strip()
    dim = read_dim(dimension, 4)
    frequency = float(content[1])

    nb_src, src_z = content[2].split()
    nb_rcv_z, rcv_z = content[3].split()
    nb_rcv_r, rcv_r = content[4].split()

    narr = int(content[5])

    i = 7
    amp = []
    phase = []
    delay_re = []
    delay_im = []
    src_ang = []
    rcv_ang = []
    nb_top_bnc = []
    nb_bot_bnc = []
    a = i + narr
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

    for x in (amp, phase, delay_re, delay_im, src_ang, rcv_ang, nb_top_bnc, nb_bot_bnc):
        if len(x) != narr:
            msg = "Lenght issue"
            raise ValueError(msg)

    arr_data = {
        "dim": dim,
        "frequency": frequency,
        "nb_src": nb_src,
        "src_z": src_z,
        "nb_rcv_z": nb_rcv_z,
        "rcv_z": rcv_z,
        "nb_rcv_r": nb_rcv_r,
        "rcv_r": rcv_r,
        "narr": narr,
        "amp": amp,
        "phase": phase,
        "delay_re": delay_re,
        "delay_im": delay_im,
        "src_ang": src_ang,
        "rcv_ang": rcv_ang,
        "nb_top_bnc": nb_top_bnc,
        "nb_bot_bnc": nb_bot_bnc,
    }

    return content, arr_data


def read_ray(file: Path, rmax: float) -> (list, dict):
    """Check the ray file created by bellhop.

    Parameters
    ----------
    file : Path
        Path to the ray file.
    rmax : float
        Distance between the source and the receiver.

    Returns
    -------
    content : list
        The contents of the ray file.

    """
    check_file_exist(file)
    check_suffix(file, ".ray")
    check_empty_file(file)
    content = file.read_text(encoding="utf-8").splitlines()

    title = content[0]
    frequency = float(content[1])

    nb_coord = content[2]
    nb_beam = content[3]

    top_depth = content[4]
    bottom_depth = content[5]
    top_d, bottom_d = read_depth(top_depth, bottom_depth)

    coord_type = content[6]
    read_coord_type(content[6])

    ra = []
    za = []
    ray_info = []
    i = 7
    while i < len(content):
        departure_angle = content[i]
        nb_steps, nb_top_ref, nb_bot_ref = content[i + 1].split()
        ray_info.append([departure_angle, nb_steps, nb_top_ref, nb_bot_ref])
        i += 2
        nb_steps = int(nb_steps)
        r = np.zeros(nb_steps)
        z = np.zeros(nb_steps)
        for nj in range(nb_steps):
            r[nj], z[nj] = content[i].split()
            i += 1
        read_r(r, rmax, nb_steps)
        ra.append(r)
        za.append(z)

    env_data = {"title": title,
                "frequency": frequency,
                "nb_coord": nb_coord,
                "nb_beam": nb_beam,
                "top": top_d,
                "bottom": bottom_d,
                "coord_type": coord_type,
                "ray_info": ray_info,
                "ra": ra,
                "za": za,
                }

    return content, env_data


def read_head_bty(header: tuple) -> None:
    """Read the header of bathymetric file."""
    if not all(check_elem_num(x, 2) for x in header):
        msg = "Wrong header in bathymetric file"
        raise ValueError(msg)


def read_env_param(line: str) -> list[float]:
    """Read the environment parameters."""
    nb_param = 4
    line = line.strip("/ ")
    if not (check_elem_num(line, nb_param)):
        msg = "Invalid environmental characteristics line"
        raise ValueError(msg)
    return [float(elem) for elem in line.split()]


def check_media(number_media: int) -> None:
    """Check the number of media."""
    if number_media != 1:
        msg = f"Invalid media line: {number_media}"
        raise ValueError(msg)


def read_depth(line: str) -> tuple[float, float]:
    """Read the depth of top and bottom."""
    depth_list = [float(elem) for elem in line.split()[1:]]
    if not all(value >= 0 for value in depth_list):
        msg = "Invalid depth line"
        raise ValueError(msg)
    if depth_list[0] >= depth_list[1]:
        msg = "Invalid depth line"
        raise ValueError(msg)
    return depth_list[0], depth_list[1]


def read_z(line: str, z_min: float) -> float:
    """Read the depth value of depth profile and compares to z_min."""
    z = float(line.split(maxsplit=1)[0])
    check_depth(z, z_min)
    return z


def read_soundspeed(line: str) -> float:
    """Read the celerity value of depth profile."""
    return float(line.split(maxsplit=2)[1])


def check_depth(z: float, z_min: float) -> None:
    """Check if depth >= minimum depth."""
    if z < z_min:
        msg = f"depth must be >= to z_min={z_min}"
        raise ValueError(msg)


def check_soundspeed_profile(d_prof: list[float]) -> None:
    """Read depth profile and check it is increasing."""
    if not all(x < y for x, y in itertools.pairwise(d_prof)):
        msg = "Depth should be increasing"
        raise ValueError(msg)


def read_bottom_properties(line: str, z_max: float) -> list[float]:
    """Read bottom properties."""
    line = line.strip(" /")
    nb_property = 6
    if not check_elem_num(line, nb_property):
        msg = "Bottom properties line: wrong number of element"
        raise ValueError(msg)
    if float(line.split(" ", maxsplit=1)[0]) != z_max:
        msg = "Bottom properties line: wrong z_max"
        raise ValueError(msg)
    return [float(elem) for elem in line.strip(" /").split()]


def check_run_type(line: str) -> None:
    """Check the run type."""
    if line not in {"E", "I", "A", "R"}:
        msg = "Incorrect run type"
        raise ValueError(msg)


def check_angle(angle: float) -> bool:
    """Check if the angle is between -180 and 180 degrees."""
    return -180 <= angle <= 180  # noqa: PLR2004


def read_angle(line: str) -> tuple[float, float]:
    """Read the beam limit angles."""
    x, y, = line.split(" ")[:2]
    x, y = float(x), float(y)
    if not all(check_angle(angle) for angle in (x, y)):
        msg = f"Invalid angle line: {line}"
        raise ValueError(msg)
    if x > y:
        msg = f"Invalid angle line: {line}"
        raise ValueError(msg)
    return x, y


def check_inf(a: float, b: float) -> bool:
    """Check if "a" is smaller than "b"."""
    return a < b


def read_coord_type(line: str) -> str:
    """Read the coordinate type."""
    if line != "'rz'":
        msg = "Invalid coordinate type"
        raise ValueError(msg)
    return line


def read_r(r: ndarray, rmax: float, nsteps: int) -> ndarray:
    """Read the range profile."""
    for nj in range(len(r)):
        r[nj] = float(r[nj])
        if not (r[nj]) >= 0:
            msg = "Invalid maximal range"
            raise ValueError(msg)
    if not all(x < y for x, y in itertools.pairwise(r)):
        msg = "Invalid range line"
        raise ValueError(msg)
    if round(r[-1], 0) != rmax:
        msg = "Invalid maximal range"
        raise ValueError(msg)
    if not (check_len_list(r, nsteps)):
        msg = "Invalid range lenght"
        raise ValueError(msg)
    return r


def read_dim(line: str, nb: int) -> str:
    """Read the dimension of calcul."""
    if not check_len_list(line, nb):
        msg = f"Invalid len of dimension line: {line}"
        raise ValueError(msg)
    dim = line[1:3]
    if dim != "2D":
        msg = f"Invalid dimension line: {dim}"
        raise ValueError(msg)
    return dim