
"""Check the files created by bellhop."""

from pathlib import Path

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
    zmin, zmax= read_depth(depth)

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

    data = {"title": title,
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

    return content,data