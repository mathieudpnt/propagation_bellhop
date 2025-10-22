
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from sub_read_env import read_depth
from utils.reader_env import read_env
from utils.sub_read_env import read_angle, read_md, read_z


def test_invalid_env_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.env")
    with pytest.raises(FileNotFoundError):
        read_env(invalid_path)

def test_invalid_env_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_env"
    (tmp_path / invalid_file).touch()
    invalid_file.write_text("test")
    with pytest.raises(ValueError, match=r"is not a .env file"):
        read_env(invalid_file)

def test_empty_env(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.env"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        read_env(empty_file)



@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("1", nullcontext(1), id="Valide media line : 1"),
        pytest.param(
            "2",
            pytest.raises(ValueError, match="Invalid media line: 2"),
            id="Too many medias",
        ),
        pytest.param(
            "-1",
            pytest.raises(ValueError, match="Invalid media line: -1"),
            id="Not enought medias",
        ),
    ],
)
def test_nb_md(line: str, expected: int) -> None:
    with expected as e:
        assert read_md(line) == e



@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("0 0.0 250.0", nullcontext((0.0, 250.0)), id="Valide depth line: 0.0 250.0"),
        pytest.param(
            "0 -15.0 250.0",
            pytest.raises(ValueError, match="Invalid depth line: 0 -15.0 250.0"),
            id="zmin should be positive",
        ),
        pytest.param(
            "0 250.0 0.0",
            pytest.raises(ValueError, match="Invalid depth line: 0 250.0 0.0"),
            id="zmin should be smaller than zmax",
        ),
        pytest.param(
            "0 0.0 0.0",
            pytest.raises(ValueError, match="Invalid depth line: 0 0.0 0.0"),
            id="zmin should be different than zmax",
        ),
    ],
)
def test_depth(line: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_depth(line) == e



@pytest.mark.parametrize(
    ("z0", "zmin", "expected"),
    [
        pytest.param(
            "0.0", 0.0, nullcontext(0.0),
            id="Valide depth line: 0.0 0.0"
        ),
        pytest.param(
            "0.0", 10.0,
            pytest.raises(ValueError, match="z0 must be equal to zmin"),
            id="z0 must be equal to zmin"
        ),
        pytest.param(
            "50.0", 0.0,
            pytest.raises(ValueError, match="z0 must be equal to zmin"),
            id="z0 must be equal to zmin"
        ),
    ],
)
def test_z(z0 : str, zmin : float, expected: float) -> None:
    with expected as e:
        assert read_z(z0, zmin) == e



@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("-120 120", nullcontext((-120.0, 120.0)), id="Valid angle "
                                                                  "line: -120 120"),
        pytest.param(
            "-120 1220",
            pytest.raises(ValueError, match="Invalid angle line: -120 1220"),
            id="zmax>180",
        ),
        pytest.param(
            "-190 120",
            pytest.raises(ValueError, match="Invalid angle line: -190 120"),
            id="zmin<180",
        ),
        pytest.param(
            "-190 190",
            pytest.raises(ValueError, match="Invalid angle line: -190 190"),
            id="zmin<180 and zmax>180",
        ),
        pytest.param(
            "120 -120",
            pytest.raises(ValueError, match="Invalid angle line: 120 -120"),
            id="pascoolraoul",
        ),
    ],
)
def test_read_angle(line: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_angle(line) == e