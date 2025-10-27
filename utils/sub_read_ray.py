def check_len(line, nb):
    line=line.split(" ")
    return len(line) == nb

def check_inf(a,b):
    return a<b

def check_pos(value):
    return value>=0

def read_depth(top, bottom):
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