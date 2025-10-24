
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