from pathlib import Path

import pytest


def read_env(file: Path) -> list[str]:
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

    return content

def get_angle(file: Path) -> tuple[float, float]:
    content = read_env(file)
    line = content[-2].strip().split()
    angle_min = float(line[0])
    angle_max = float(line[1])

    for angle in (angle_min, angle_max):
        if not -180 <= angle <= 180:  # noqa: PLR2004
            msg = "angle must be between -180 and 180"
            raise ValueError(msg)
    return float(line[0]), float(line[1])


def test_angle(sample_env: tuple[Path, str]) -> None:
    try:
        angles = get_angle(sample_env)
    except ValueError as e:
        assert str(e) == "angle must be between -180 and 180"
    else:
        # If no exception, assert the angles are valid
        assert isinstance(angles, tuple)
        assert len(angles) == 2
        assert all(-180 <= a <= 180 for a in angles)

def test_invalid_env_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.env")
    with pytest.raises(FileNotFoundError):
        read_env(invalid_path)


def test_invalid_env_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_env"
    invalid_file.write_text("some content")
    with pytest.raises(ValueError, match=r"is not a .env file"):
        read_env(invalid_file)


def test_empty_env(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.env"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        read_env(empty_file)
