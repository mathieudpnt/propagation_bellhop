
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from utils.reader_arr import read_arr
from utils.sub_read_arr import read_dim, read_src_angle, check_len


def test_invalid_arr_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.arr")
    with pytest.raises(FileNotFoundError):
        read_arr(invalid_path)

def test_invalid_env_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_arr"
    (tmp_path / invalid_file).touch()
    invalid_file.write_text("test")
    with pytest.raises(ValueError, match=r"is not a .arr file"):
        read_arr(invalid_file)

def test_empty_env(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.arr"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        read_arr(empty_file)

@pytest.mark.parametrize(
    ("line", "nb", "expected"),
    [
        pytest.param(
            "'2D'",
            4,
            nullcontext("2D"),
            id="Valide dimension",
        ),
        pytest.param(
            "'2D a'",
            4,
            pytest.raises(ValueError, match="Invalid len of dimension line: '2D a'"),
            id="Invalid len of dimension line",
        ),
        pytest.param(
            "'3D'",
            4,
            pytest.raises(ValueError, match="Invalid dimension line: 3D"),
            id="Dimension should be 2D",
        ),
    ],
)
def test_read_dim(line: str, nb : int, expected: list[str]) -> None:
    with expected as e:
        assert read_dim(line, nb) == e

@pytest.mark.parametrize(
    ("list", "expected"),
    [
        pytest.param(
            [4, 3, 2, 1],
            nullcontext([4, 3, 2, 1]),
            id="Valide source angle list",
        ),
        pytest.param(
            [1, 2, 3, 4],
            pytest.raises(ValueError, match="Invalid source angle list"),
            id="Depth should be decreasing",
        ),
        pytest.param(
            [1, 5, 3, 1],
            pytest.raises(ValueError, match="Invalid source angle list"),
            id="Depth should be decreasing",
        ),
    ],
)
def test_read_src_ang(list: list[float], expected: list[str]) -> None:
    with expected as e:
        assert read_src_angle(list) == e

