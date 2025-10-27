from pathlib import Path

import numpy as np


def read_env(file: Path) -> (list, dict):
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
    if file.suffix != ".env":
        msg = f"{file} is not a .env file"
        raise ValueError(msg)

    if not file.exists():
        msg = f"{file} does not exist"
        raise FileNotFoundError(msg)

    content = file.read_text().splitlines()

    if not content:
        msg = f"{file} is empty"
        raise ValueError(msg)

    title = content[0]
    frequency=int(content[1])

    number_media=content[2]
    number_media=read_md(number_media)

    env_opt=content[3]

    env_param=content[4]
    env_param=read_env_param(env_param,4)

    depth=content[5]
    zmin, zmax= read_ray_depth(depth)

    depth_prof,sound_speed_prof = content [6].split(" ")[:2]
    z0=read_z(depth_prof, zmin)

    depth_prof, sound_speed_prof = [float(depth_prof)], [float(sound_speed_prof)]
    i=7
    while float(content[i].split(" ")[0])<zmax:
        depth_prof.append(float(content[i].split(" ")[0]))
        sound_speed_prof.append(float(content[i].split(" ")[1]))
        i += 1
    z_fin, c_fin = (content[i].split(" ")[:2])
    z_fin, c_fin = float(z_fin), float(c_fin)
    check_eq(z_fin, zmax)
    depth_prof.append(z_fin)
    sound_speed_prof.append(c_fin)
    d_prof=read_prof(depth_prof)

    bot_cond=content[i+1]
    bot_prop=content[i+2]
    b_prop=read_bot_prop(bot_prop,7, zmax)

    nb_src=content[i+3]
    src_z=content[i+4]
    nb_rcv_z=content[i+5]
    rdv_z=content[i+6]
    nb_rcv_r = content[i+7]
    rdv_r = content[i+8]

    run_type= content[i+9]
    r_type=read_run_type(run_type)

    nb_beam = content[i + 10]

    ang = content[i + 11]
    x, y = read_angle(ang)

    info = content[i + 12]

    env_data = {"title": title,
            "frequency": frequency,
            "number_media" : number_media,
            "env_opt" : env_opt,
            "env_param" : env_param,
            "zmin, zmax" : (zmin, zmax),
            "z0" : z0,
            "d_prof" : d_prof,
            "sound_speed_prof" : sound_speed_prof,
            "bot_cond" : bot_cond,
            "b_prop" : b_prop,
            "nb_src" : nb_src,
            "src_z" : src_z,
            "nb_rcv_z" : nb_rcv_z,
            "rdv_z" : rdv_z,
            "nb_rcv_r" : nb_rcv_r,
            "rdv_r" : rdv_r,
            "r_type" : r_type,
            "nb_beam" : nb_beam,
            "angles" : (x,y),
            "info" : info,
            }

    return content,env_data


def read_arr(file: Path) -> list :
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

    content = [elem.strip() for elem in file.read_text().splitlines()]

    if not content:
        msg = f"{file} is empty"
        raise ValueError(msg)

    dimension = content[0].strip()
    dim=read_dim(dimension,2)
    frequency=float(content[1])

    nb_src, src_z =content[2].split()
    nb_rcv_z, rcv_z=content[3].split()
    nb_rcv_r, rcv_r=content[4].split()

    narr=int(content[5])
    narr=int(content[6])

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

    for x in (amp,phase,delay_re,delay_im,src_ang,rcv_ang,nb_top_bnc,nb_bot_bnc):
        check_len(x,narr)

    arr_data = {
        "dim": dim,
        "frequency": frequency,
        "nb_src": nb_src,
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

    return content,arr_data


def read_ray(file: Path, rmax) -> (list, dict):
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
    if file.suffix != ".ray":
        msg = f"{file} is not a .ray file"
        raise ValueError(msg)

    if not file.exists():
        msg = f"{file} does not exist"
        raise FileNotFoundError(msg)

    content = [elem.strip() for elem in file.read_text().splitlines()]

    if not content:
        msg = f"{file} is empty"
        raise ValueError(msg)

    title = content[0]
    frequency=float(content[1])

    nb_coord = content[2]
    nb_beam = content[3]

    top_depth = content[4]
    bottom_depth = content[5]
    top, bottom=read_ray_depth(top_depth, bottom_depth)

    coord_type = content [6]
    read_coord_type(coord_type)

    ra=[]
    za=[]
    ray_info=[]
    i=7
    while i<len(content):
        departure_angle = content[i]
        nb_steps, nb_top_ref, nb_bot_ref = content[i+1].split()
        ray_info.append([departure_angle, nb_steps, nb_top_ref, nb_bot_ref])
        i+=2
        nb_steps = int(nb_steps)
        r = np.zeros(nb_steps)
        z = np.zeros(nb_steps)
        for nj in range(nb_steps):
            r[nj], z[nj] = content[i].split()
            i+=1
        read_r(r, rmax, nb_steps)
        ra.append(r)
        za.append(z)
        # assert 0<=round(r[nj],0)<=rmax
        # assert len( r ) == nsteps
        # assert round(r[-1],0)==rmax

    env_data = {"title" : title,
                "frequency" : frequency,
                "nb_coord" : nb_coord,
                "nb_beam" : nb_beam,
                "top_depth" : top_depth,
                "bottom_depth" : bottom_depth,
                "coord_type" : coord_type,
                "ray_info" : ray_info,
                "ra" :ra,
                "za" : za,
                }

    return content,env_data


def check_len(line, nb):
    line=line.split(" ")
    return len(line) == nb


def check_pos(value):
    return value>=0


def check_eq(a,b):
    return a==b


def read_env_param(line, nb):
    if not (check_len(line,nb)):
        msg="Invalid env_carac line"
        raise ValueError(msg)
    return line.split(" ")


def read_md(number_media):
    number_media = int(number_media)
    if not check_eq(number_media,1):
        msg=f"Invalid media line: {number_media}"
        raise ValueError(msg)
    return number_media


def read_depth(line):
    (zmin,zmax) = line.split(" ")[1:]
    zmin, zmax = [float(zmin), float(zmax)]
    if not all(check_pos(value) for value in (zmin, zmax)):
        msg=f"Invalid depth line: {line}"
        raise ValueError(msg)
    if zmin>=zmax:
        msg=f"Invalid depth line: {line}"
        raise ValueError(msg)
    return zmin, zmax


def read_z(z0, zmin):
    z0=float(z0)
    if not check_eq(z0, zmin):
        msg="z0 must be equal to zmin"
        raise ValueError(msg)
    return z0


def read_prof(d_prof):
    if not all(x < y for x, y in zip(d_prof, d_prof[1:])):
        msg="Depth should be increasing"
        raise ValueError(msg)
    return d_prof


def read_bot_prop(line, nb, zmax):
    if not (check_len(line,nb)):
        msg=f"Invalid len bot_prop line"
        raise ValueError(msg)
    if not check_eq(float(line.split(" ")[0]),zmax):
        msg=f"Invalid bot_prop line"
        raise ValueError(msg)
    return line.split(" ")[:-1]


def read_run_type(line):
    if line not in {"E", "I", "A", "R"}:
        msg = "Incorrect run type"
        raise ValueError(msg)
    return line


def check_angle(angle: float) -> bool:
    return -180 <= angle <= 180


def read_angle(line: str) -> tuple[float, float]:
    x,y,=line.split(" ")[:2]
    x,y = float(x), float(y)
    if not all(check_angle(angle) for angle in (x,y)):
        msg=f"Invalid angle line: {line}"
        raise ValueError(msg)
    if x>y:
        msg=f"Invalid angle line: {line}"
        raise ValueError(msg)
    return x,y


def check_inf(a,b):
    return a<b


def read_ray_depth(top, bottom):
    top, bottom=float(top),float(bottom)
    if not all (check_pos(value) for value in (top, bottom)):
        msg = "Invalid depth line"
        raise ValueError(msg)
    if not check_inf(top, bottom):
        msg = "Invalid depth line"
        raise ValueError(msg)
    return top, bottom


def read_coord_type(line):
    if line != "'rz'":
        msg="Invalid coordinate type"
        raise ValueError(msg)
    return line


def read_r(r, rmax, nsteps):
    for nj in range(len(r)):
        r[nj]=float(r[nj])
        if not (r[nj]) >= 0:
            msg = "Invalid maximal range"
            raise ValueError(msg)
    if not all(x < y for x, y in zip(r, r[1:])):
        msg="Invalid range line"
        raise ValueError(msg)
    if round(r[-1],0) != rmax:
        msg = "Invalid maximal range"
        raise ValueError(msg)
    if not (check_len(str(r), nsteps)):
        msg = "Invalid range lenght"
        raise ValueError(msg)
    return r


def read_dim(line, nb):
    if not check_len(line, nb):
        msg = f"Invalid len of dimension line: {line}"
        raise ValueError(msg)
    dim=line[1:3]
    if dim != '2D':
        msg=f"Invalid dimension line: {dim}"
        raise ValueError(msg)
    return dim


def read_src_angle(list):
    if not all(x > y for x, y in zip(list, list[1:])):
        msg="Invalid source angle list"
        raise ValueError(msg)
    return list
