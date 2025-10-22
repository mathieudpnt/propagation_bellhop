
"""Check the files created by bellhop."""

from pathlib import Path

from sub_read_env import read_angle

from utils.sub_read_env import check_angle


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
    number_media=int(content[2])
    env_opt=content[3]
    env_carac=content[4]
    depth=content[5]
    depth_prof,sound_speed_prof = content [6].split(" ")[:2]
    depth_prof, sound_speed_prof = [float(depth_prof)], [float(sound_speed_prof)]
    i=7
    while float(content[i].split(" ")[0])<float(depth.split(" ")[2]):
        depth_prof.append(float(content[i].split(" ")[0]))
        sound_speed_prof.append(float(content[i].split(" ")[1]))
        i+=1
    zmax=float(content[i].split(" ")[0])
    bot_cond=content[i+1]
    bot_prop=content[i+2]

    nb_src=content[i+3]
    src_z=content[i+4]
    nb_rcv_z=content[i+5]
    rdv_z=content[i+6]
    nb_rcv_r = content[i+7]
    rdv_r = content[i+8]
    run_type= content[i+9]
    nb_beam=content[i+10]
    ang= content[i+11]
    read_angle(ang)

    info= content[i+12]

    return content


