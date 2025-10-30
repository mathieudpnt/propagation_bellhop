from contextlib import nullcontext
from pathlib import Path

import pytest
from numpy import ndarray
from reader_utils import (
    check_empty_file,
    f_exist,
    invalid_suffix,
    read_coord_type,
    read_depth,
    read_r,
)


def test_invalid_ray_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.ray")
    with pytest.raises(FileNotFoundError):
        f_exist(invalid_path)


def test_invalid_ray_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_ray"
    (tmp_path / invalid_file).touch()
    invalid_file.write_text("test")
    with pytest.raises(ValueError, match=r"is not a .ray file"):
        invalid_suffix(invalid_file, ".ray")


def test_empty_ray(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.ray"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        check_empty_file(empty_file)


@pytest.mark.parametrize(
    ("top", "bottom", "expected"),
    [
        pytest.param(
            "0.0", "250",
            nullcontext((0.0, 250)),
            id="Valide depth",
        ),
        pytest.param(
            "0.0", "-250",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="Depth must be positive",
        ),
        pytest.param(
            "250", "0.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="Depth must be increasing",
        ),
    ],
)
def test_read_depth(top: float, bottom: float, expected: tuple) -> None:
    with expected as e:
        assert read_depth(top, bottom) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            "'rz'",
            nullcontext("'rz'"),
            id="Valide depth",
        ),
        pytest.param(
            "'r'",
            pytest.raises(ValueError, match="Invalid coordinate type"),
            id="Coordinate type must be 'rz'",
        ),
        pytest.param(
            "250",
            pytest.raises(ValueError, match="Invalid coordinate type"),
            id="Coordinate type must be str",
        ),
    ],
)
def test_read_coord(line: str, expected: str) -> None:
    with expected as e:
        assert read_coord_type(line) == e


@pytest.mark.parametrize(
    ("r", "rmax", "nsteps", "expected"),
    [
        pytest.param(
            [1, 2, 5, 1781],
            1781,
            4,
            nullcontext([1, 2, 5, 1781]),
            id="Valide depth",
        ),
        pytest.param(
            [1, 2, 5, 9],
            1781,
            4,
            pytest.raises(ValueError, match="Invalid maximal range"),
            id="Maximal range must be rmax",
        ),
        pytest.param(
            [1, 2, 1781, 9],
            1781,
            4,
            pytest.raises(ValueError, match="Invalid range line"),
            id="range must be increasing",
        ),
        pytest.param(
            [1, 2, 1781],
            1781,
            4,
            pytest.raises(ValueError, match="Invalid range lenght"),
            id="r must contain nsteps values",
        ),
    ],
)
def test_read_r(r: ndarray, rmax: float, nsteps: str, expected: str) -> None:
    with expected as e:
        assert read_r(r, rmax, nsteps) == e
