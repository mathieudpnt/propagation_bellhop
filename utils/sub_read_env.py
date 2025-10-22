
def check_len(line, nb):
    line=line.split(" ")
    return len(line) == nb

def check_pos(value):
    return value>=0

def check_diff(a,b):
    return a!=b

def read_env_param(line, nb):
    if not (check_len(line,nb)):
        raise ValueError(f"Invalid env_carac line: {line}")
    return line.split(" ")

def read_md(number_media):
    number_media = int(number_media)
    if check_diff(number_media,1):
        raise ValueError(f"Invalid media line: {number_media}")
    return number_media

def read_depth(line):
    (zmin,zmax) = line.split(" ")[1:]
    zmin, zmax = [float(zmin), float(zmax)]
    if not all(check_pos(value) for value in (zmin, zmax)):
        raise ValueError(f"Invalid depth line: {line}")
    if zmin>=zmax:
        raise ValueError(f"Invalid depth line: {line}")
    return zmin, zmax

def read_z(z0, zmin):
    z0=float(z0)
    if check_diff(z0, zmin):
        raise ValueError("z0 must be equal to zmin")
    return z0

def read_prof(d_prof):
    if not all(x < y for x, y in zip(d_prof, d_prof[1:])):
        raise ValueError("depth should be increasing")
    return d_prof

def check_angle(angle: float) -> bool:
    return -180 <= angle <= 180

def read_angle(line: str) -> tuple[float, float]:
    x,y,=line.split(" ")[:2]
    x,y = float(x), float(y)
    if not all(check_angle(angle) for angle in (x,y)):
        raise ValueError(f"Invalid angle line: {line}")
    if x>y:
        raise ValueError(f"Invalid angle line: {line}")
    return x,y


