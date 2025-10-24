
"""Check the files created by bellhop."""

from pathlib import Path
import numpy as np


def read_env(file: Path) -> (list, dict):
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
    coord_type = content [6]

    ra=[]
    za=[]
    dep_ang=[]
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
