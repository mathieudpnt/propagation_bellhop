
from contextlib import nullcontext
from pathlib import Path

import pytest
from reader_utils import (
    check_empty_file,
    f_exist,
    check_suffix,
    read_angle,
    read_bot_prop,
    read_depth,
    read_env_param,
    read_md,
    read_prof,
    read_run_type,
    read_z,
)


def test_invalid_env_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.env")
    with pytest.raises(FileNotFoundError):
        f_exist(invalid_path)


def test_invalid_env_suffix(tmp_path: Path) -> None:
    invalid_file = tmp_path / "file.not_env"
    (tmp_path / invalid_file).touch()
    invalid_file.write_text("test")
    with pytest.raises(ValueError, match=r"is not a .env file"):
        check_suffix(invalid_file, ".env")


def test_empty_env(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.env"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        check_empty_file(empty_file)


@pytest.mark.parametrize(
    ("line", "nb", "expected"),
    [
        pytest.param(
            "1 2 a bc /",
            5,
            nullcontext(["1", "2", "a", "bc", "/"]),
            id="Valid number of values",
        ),
        pytest.param(
            "2 abc /",
            2,
            pytest.raises(ValueError,
                          match="Invalid environmental characteristics line"),
            id="Too many values",
        ),
        pytest.param(
            "-1 /",
            3,
            pytest.raises(ValueError,
                          match="Invalid environmental characteristics line"),
            id="Not enought medias",
        ),
    ],
)
def test_env_param(line: str, nb: int, expected: list[str]) -> None:
    with expected as e:
        assert read_env_param(line, nb) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("1", nullcontext(1), id="Valid media line : 1"),
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
    ("a", "b", "expected"),
    [
        pytest.param("0.0", "250.0", nullcontext((0.0, 250.0)), id="Valid depth line"),
        pytest.param(
            "-15.0", "250.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="zmin should be positive",
        ),
        pytest.param(
            "250.0", "0.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="zmin should be smaller than zmax",
        ),
        pytest.param(
            "0.0", "0.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="zmin should be different than zmax",
        ),
    ],
)
def test_depth(a: str, b: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_depth(a, b) == e


@pytest.mark.parametrize(
    ("z0", "zmin", "expected"),
    [
        pytest.param(
            "0.0", 0.0, nullcontext(0.0),
            id="Valid depth line: 0.0 0.0",
        ),
        pytest.param(
            "0.0", 10.0,
            pytest.raises(ValueError, match="z0 must be equal to zmin"),
            id="z0 must be equal to zmin",
        ),
        pytest.param(
            "50.0", 0.0,
            pytest.raises(ValueError, match="z0 must be equal to zmin"),
            id="z0 must be equal to zmin",
        ),
    ],
)
def test_z(z0: str, zmin: float, expected: float) -> None:
    with expected as e:
        assert read_z(z0, zmin) == e


@pytest.mark.parametrize(
    ("d_prof", "expected"),
    [
        pytest.param([1, 2, 3, 4], nullcontext([1, 2, 3, 4]), id="Valid depth profil "),
        pytest.param(
            [6, 3, 2, 0],
            pytest.raises(ValueError, match="Depth should be increasing"),
            id="decreasing depth profil line",
        ),
        pytest.param(
            [5, 5, 5, 5],
            pytest.raises(ValueError, match="Depth should be increasing"),
            id="stable depth profil line",
        ),
        pytest.param(
            [1, 4, 3, 9],
            pytest.raises(ValueError, match="Depth should be increasing"),
            id="unstable depth profil line"),
    ],
)
def test_read_prof(d_prof: list[float, float, float, float],
                   expected: list[float]) -> None:
    with expected as e:
        assert read_prof(d_prof) == e


@pytest.mark.parametrize(
    ("line", "nb", "zmax", "expected"),
    [
        pytest.param(
            "250 1600.0 0.0 1.75 1.05 0.0 /",
            7,
            250,
            nullcontext(["250", "1600.0", "0.0", "1.75", "1.05", "0.0"]),
            id="Valid bot_prop line",
        ),
        pytest.param(
            "250 1600.0 0.0 1.75 1.05", 7, 250,
            pytest.raises(ValueError, match="Invalid len bot_prop line"),
            id="Bottom parameter value missing",
        ),
        pytest.param(
                "250 1600.0 0.0 1.75 1.05 0.0 / /", 7, 250,
                pytest.raises(ValueError, match="Invalid len bot_prop line"),
                id="Too many botom parameters",
        ),
        pytest.param(
            "100 1600.0 0.0 1.75 1.05 0.0 /", 7, 250,
            pytest.raises(ValueError, match="Invalid bot_prop line"),
            id="Depth must be zmax",
        ),
    ],
)
def test_read_bot_prop(line: str, nb: int, zmax: float, expected: list[float]) -> None:
    with expected as e:
        assert read_bot_prop(line, nb, zmax) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("E", nullcontext("E"), id="Valid run type"),
        pytest.param(
            "M",
            pytest.raises(ValueError, match="Incorrect run type"),
            id="Incorrect run type",
        ),
        pytest.param(
            "EA",
            pytest.raises(ValueError, match="Incorrect run type"),
            id="Incorrect run type",
        ),
        pytest.param(
            "5",
            pytest.raises(ValueError, match="Incorrect run type"),
            id="Incorrect run type"),
    ],
)
def test_read_run_type(line: str, expected: str) -> None:
    with expected as e:
        assert read_run_type(line) == e


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
            id="zmin>zmax",
        ),
    ],
)
def test_read_angle(line: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_angle(line) == e
