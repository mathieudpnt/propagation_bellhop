
from contextlib import nullcontext
from pathlib import Path

import pytest

from utils.reader_env import read_env
from utils.sub_read_env import read_angle


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
        pytest.param("-120 120", nullcontext((-120.0, 120.0)), id="zmin<zmax"),
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
    ],
)
def test_read_angle(line: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_angle(line) == e