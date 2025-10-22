

def check_angle(angle: float) -> bool:
    return -180 <= angle <= 180

def read_angle(line: str) -> tuple[float, float]:
    x,y,=line.split(" ")[:2]
    x,y = float(x), float(y)
    if not all(check_angle(angle) for angle in (x,y)):
        raise ValueError(f"Invalid angle line: {line}")
    return x,y
