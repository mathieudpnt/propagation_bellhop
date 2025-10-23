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

def check_len(list, nb):
    return len(list) == nb