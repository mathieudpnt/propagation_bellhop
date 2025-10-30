
from pathlib import Path

import pytest

from utils.reader_utils import f_exist, invalid_suffix, check_empty_file


def test_invalid_bty_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.bty")
    with pytest.raises(FileNotFoundError):
        f_exist(invalid_path)


def test_invalid_bty_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_bty"
    (tmp_path / invalid_file).touch()
    invalid_file.write_text("test")
    with pytest.raises(ValueError, match=r"is not a .bty file"):
        invalid_suffix(invalid_file, ".bty")


def test_empty_bty(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.bty"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        check_empty_file(empty_file)