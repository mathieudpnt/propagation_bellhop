
from contextlib import nullcontext
from pathlib import Path

import pytest
from reader_utils import (
    check_empty_file,
    f_exist,
    invalid_suffix,
    read_dim,
)


def test_invalid_arr_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.arr")
    with pytest.raises(FileNotFoundError):
        f_exist(invalid_path)


def test_invalid_arr_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_arr"
    (tmp_path / invalid_file).touch()
    invalid_file.write_text("test")
    with pytest.raises(ValueError, match=r"is not a .arr file"):
        invalid_suffix(invalid_file, ".arr")


def test_empty_arr(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.arr"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        check_empty_file(empty_file)


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
def test_read_dim(line: str, nb: int, expected: list[str]) -> None:
    with expected as e:
        assert read_dim(line, nb) == e
