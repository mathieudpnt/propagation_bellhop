"""Read files."""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from core_utils import (
    check_elem_num,
    check_empty_file,
    check_file_exist,
    check_len_list,
    check_suffix, zeros
)
from numpy import ndarray

if TYPE_CHECKING:
    from pathlib import Path


def read_bellhop_file(file: Path) -> list[str]:
    """Read a Bellhop file.

    Read .env/.arr/.bty files produced by Bellhop.

    Parameters
    ----------
    file : Path
        Path to the file.

    Returns
    -------
    content : list[str]
        The content of the environmental file.

    """
    if file.suffix not in {".env", ".arr", ".bty", ".ray"}:
        msg = f"{file} is not a Bellhop file."
        raise ValueError(msg)
    check_file_exist(file)
    check_empty_file(file)
    return [
        elem.strip(" /").strip("'")
        for elem in file.read_text(encoding="utf-8").splitlines()
    ]


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
    check_suffix(file, ".env")
    content = read_bellhop_file(file)

    title = content[0]
    frequency = int(content[1])
    number_media = read_media(content[2])
    env_opt = content[3]
    temperature, salinity, ph, pressure = read_env_param(content[4])
    z_min, z_max = read_depth_line(content[5])

    depth_profile, sound_speed_profile, idx = read_profile(content)

    bottom_cond = content[idx + 1]
    bottom_prop = read_bottom_properties(content[idx + 2], z_max)

    nb_src = int(content[idx + 3])
    src_z = float(content[idx + 4])
    nb_rcv_z = int(content[idx + 5])

    if nb_rcv_z == 1:
        rcv_z = float(content[idx + 6])
    else:
        rcv_z = tuple(float(elem) for elem in content[idx + 6].split())

    nb_rcv_r = int(content[idx + 7])

    if nb_rcv_z == 1:
        rcv_r = float(content[idx + 8])
    else:
        rcv_r = tuple(float(elem) for elem in content[idx + 8].split())

    run_type = read_run_type(content[idx + 9])
    nb_beam = int(content[idx + 10])
    angle_inf, angle_sup = read_angle(content[idx + 11])
    info = content[idx + 12]

    return {
        "title": title,
        "frequency": frequency,
        "number_media": number_media,
        "env_opt": env_opt,
        "temperature": temperature,
        "salinity": salinity,
        "ph": ph,
        "pressure": pressure,
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
        "info": info,
    }


def read_arr(file: Path) -> dict:
    """Read the .asc file resulting from using Bellhop.

    and extracts the ray's arrival times
    Only runs if the arrival calculation type (a) is chosen

    Parameters
    ----------
    file : Path
        Path and name of the .asc file to read

    Returns
    -------
    arr : ndarray
        Array that contains information about the rays path
        (total number of studied rays, wave equation,
        complex delay, departure angle, arrival angle,
        number of top reflexions, number of bottom reflexions)
    content : list
        Other information in the file.

    """
    check_suffix(file, ".arr")
    content = read_bellhop_file(file)

    dim = read_dimension(content[0])
    frequency = float(content[1])
    nb_src, src_z = read_nb_depth_range(content[2])
    nb_rcv_z, rcv_z = read_nb_depth_range(content[3])
    nb_rcv_r, rcv_r = read_nb_depth_range(content[4])
    nb_arr = read_nb_ray(content[5])

    i = 7
    wave_eq = zeros(nb_arr, 1)  # wave equation
    delay = zeros(nb_arr, 1)  # complex delay
    delay_re = zeros(nb_arr, 1)
    delay_im = zeros(nb_arr, 1)
    src_ang = zeros(nb_arr, 0)  # departure angle
    rcv_ang = zeros(nb_arr, 0)  # arrival angle
    nb_top_bnc = []  # number of top reflexions
    nb_bot_bnc = []

    a = i + nb_arr
    k = 0
    while i < a:
        line = content[i].split()
        amp = (float(line[0]))
        phase = (float(line[1]))
        delay_re[k] = (float(line[2]))
        delay_im[k] = (float(line[3]))
        src_ang[k] = (float(line[4]))
        rcv_ang[k] = (float(line[5]))
        nb_top_bnc.append(int(line[6]))
        nb_bot_bnc.append(int(line[7]))
        wave_eq[k] = (amp * np.exp(1j * phase * np.pi / 180.0))  # complex wave equation
        delay[k] = (delay_re[k] + 1j * delay_im[k])  # complex delay
        k += 1
        i += 1

    if not all(len(arr) == nb_arr for arr in (
            delay_re,
            delay_im,
            src_ang,
            rcv_ang,
            nb_top_bnc,
            nb_bot_bnc,
    )):
        msg = "Inconsistent length"
        raise ValueError(msg)

    arr = {"nb_arr": nb_arr,
           "wave_eq": wave_eq,
           "delay": delay,
           "src_angle": src_ang,
           "rcv_angle": rcv_ang,
           "nb_top_bnc": nb_top_bnc,
           "nb_bot_bnc": nb_bot_bnc}

    return arr, {"dim": dim,
        "frequency": frequency,
        "nb_src": nb_src,
        "src_z": src_z,
        "nb_rcv_z": nb_rcv_z,
        "rcv_z": rcv_z,
        "nb_rcv_r": nb_rcv_r,
        "rcv_r": rcv_r}



def read_ray(file: Path) -> dict:
    """Check the ray file created by bellhop.

    Parameters
    ----------
    file : Path
        Path to the ray file.

    Returns
    -------
    content : list
        The contents of the ray file.

    """
    check_suffix(file, ".ray")
    content = read_bellhop_file(file)

    title = content[0].strip()
    frequency = float(content[1])
    nb_coord = content[2].strip()
    nb_beam = content[3].strip()
    top_depth = float(content[4])
    bottom_depth = float(content[5])
    coord_type = read_coord_type(content[6]).strip()

    ra = []
    za = []
    ray_info = []
    i = 7
    while i < len(content):
        departure_angle = float(content[i])
        nb_steps, nb_top_ref, nb_bot_ref = tuple(int(elem)
                                                 for elem in content[i + 1].split())
        ray_info.append([departure_angle, nb_steps, nb_top_ref, nb_bot_ref])
        i += 2
        r = np.zeros(nb_steps)
        z = np.zeros(nb_steps)
        for step in range(nb_steps):
            r[step], z[step] = content[i].split()
            r[step] = float(r[step])
            z[step] = float(z[step])
            i += 1
        read_r(r, nb_steps)
        ra.append(r)
        za.append(z)

    return {"title": title,
                "frequency": frequency,
                "nb_coord": nb_coord,
                "nb_beam": nb_beam,
                "top": top_depth,
                "bottom": bottom_depth,
                "coord_type": coord_type,
                "ray_info": ray_info,
                "ra": ra,
                "za": za,
    }


def read_profile(content: list[str]) -> tuple[list[float], list[float], int]:
    """Read depth and sound speed profiles of .env file."""
    _, z_max = read_depth_line(content[5])

    i = 6
    depth_profile = []
    sound_speed_profile = []
    while True:
        depth_profile.append(read_profile_depth_value(content[i]))
        sound_speed_profile.append(read_profile_soundspeed_value(content[i]))
        if depth_profile[-1] >= z_max:
            break
        i += 1

    # verify last value
    if depth_profile[-1] != z_max:
        msg = "Wrong last depth value"
        raise ValueError(msg)

    check_soundspeed_profile(depth_profile)

    return depth_profile, sound_speed_profile, i


def read_nb_depth_range(line: str) -> tuple[int, float]:
    """Read number and depth of point in .arr file.

    This applies for source/receiver/range lines.
    """
    try:
        split = line.split(maxsplit=1)
        nb_src = int(split[0])
        depth_src = float(split[1])
    except ValueError:
        msg = "Invalid line"
        raise ValueError(msg) from None
    return nb_src, depth_src


def read_nb_ray(line: str) -> int:
    """Read ray number in .arr file."""
    try:
        return int(line)
    except ValueError:
        msg = "Invalid line"
        raise ValueError(msg) from None


def read_head_bty(header: tuple) -> None:
    """Read the header of bathymetric file."""
    if not all(check_elem_num(x, 2) for x in header):
        msg = "Wrong header in bathymetric file"
        raise ValueError(msg)


def read_env_param(line: str) -> tuple[float, float, float, float]:
    """Read the environment parameters."""
    nb_param = 4
    if not (check_elem_num(line, nb_param)):
        msg = "Invalid environmental characteristics line"
        raise ValueError(msg)
    temp, sal, ph, pressure = tuple(float(elem) for elem in line.split())
    return temp, sal, ph, pressure


def read_media(media_line: str) -> int:
    """Read and validate the number of media."""
    try:
        media = int(media_line)
    except ValueError:
        msg = "Wrong media format"
        raise ValueError(msg) from None

    if media != 1:
        msg = "Wrong media number"
        raise ValueError(msg)
    return media


def read_depth_line(line: str) -> tuple[float, float]:
    """Read the min and max depth in .env file."""
    depth_list = [float(elem) for elem in line.split()[1:]]
    if not all(value >= 0 for value in depth_list):
        msg = "Invalid depth line"
        raise ValueError(msg)
    if depth_list[0] >= depth_list[1]:
        msg = "Invalid depth line"
        raise ValueError(msg)
    return depth_list[0], depth_list[1]


def read_profile_depth_value(line: str) -> float:
    """Read the depth value of depth profile in .env file."""
    return float(line.split(maxsplit=1)[0])


def read_profile_soundspeed_value(line: str) -> float:
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
    nb_property = 6
    if not check_elem_num(line, nb_property):
        msg = "Bottom properties line: wrong number of element"
        raise ValueError(msg)
    if float(line.split(" ", maxsplit=1)[0]) != z_max:
        msg = "Bottom properties line: wrong z_max"
        raise ValueError(msg)
    return [float(elem) for elem in line.split()]


def read_run_type(line: str) -> str:
    """Check the run type."""
    if line not in {"E", "I", "A", "R"}:
        msg = "Incorrect run type"
        raise ValueError(msg)
    return line


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
    if line != "rz":
        msg = "Invalid coordinate type"
        raise ValueError(msg)
    return line


def read_r(r: ndarray, n_steps: int) -> ndarray:
    """Read the range profile."""
    for nj in range(len(r)):
        r[nj] = float(r[nj])
        if r[nj] < 0:
            msg = "Invalid maximal range: negative value"
            raise ValueError(msg)
    if not all(x <= y for x, y in itertools.pairwise(r)):
        msg = "Invalid range line: non increasing"
        raise ValueError(msg)
    if not (check_len_list(r, n_steps)):
        msg = "Invalid range length: wrong length"
        raise ValueError(msg)
    return r


def read_dimension(line: str) -> int:
    """Read the dimension in .arr file."""
    try:
        dim = int(line.strip("D"))
    except ValueError:
        msg = "Invalid dimension line"
        raise ValueError(msg) from None
    if dim != 2:
        msg = f"Invalid dimension number: {dim}"
        raise ValueError(msg)
    return dim