
from contextlib import nullcontext
from pathlib import Path

import pytest
from reader_utils import (
    read_angle,
    read_bottom_properties,
    read_depth,
    read_env_param,
    check_media,
    check_soundspeed_profile,
    check_run_type,
    read_z,
)
from core_utils import check_file_exist, check_suffix, check_empty_file


def test_invalid_env_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist.env")
    with pytest.raises(FileNotFoundError):
        check_file_exist(invalid_path)


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
            "1 2 3 /",
            4,
            nullcontext([1, 2, 3]),
            id="valid entry",
        ),
        pytest.param(
            "1 2 3 4 /",
            2,
            pytest.raises(ValueError,
                          match="Invalid environmental characteristics line"),
            id="too many values",
        ),
        pytest.param(
            "1 2 /",
            3,
            pytest.raises(ValueError,
                          match="Invalid environmental characteristics line"),
            id="not enough values",
        ),
    ],
)
def test_env_param(line: str, nb: int, expected: list[str]) -> None:
    with expected as e:
        assert read_env_param(line) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(1,
                     nullcontext(),
                     id="valid entry"
        ),
        pytest.param(
            2,
            pytest.raises(ValueError, match="Invalid media line: 2"),
            id="too many medias",
        ),
        pytest.param(
            0,
            pytest.raises(ValueError, match="Invalid media line: 0"),
            id="not enough medias",
        ),
    ],
)
def test_nb_md(line: int, expected: int) -> None:
    with expected as e:
        assert check_media(line) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            "0 0.0 250",
            nullcontext((0, 250)),
            id="valid entry"
        ),
        pytest.param(
            "0 -15.0 250.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="negative z_min",
        ),
        pytest.param(
            "0 250.0 0.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="z_min > z_max",
        ),
        pytest.param(
            "0 0.0 0.0",
            pytest.raises(ValueError, match="Invalid depth line"),
            id="z_min = z_max",
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
            id="Valid depth line: 0.0 0.0",
        ),
        pytest.param(
            "0.0", 10.0,
            pytest.raises(ValueError, match=r"depth must be >= to z_min=10.0"),
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
        pytest.param([1, 2, 3, 4], nullcontext(), id="valid depth profile"),
        pytest.param(
            [6, 3, 2, 0],
            pytest.raises(ValueError, match="Depth should be increasing"),
            id="decreasing depth profile line",
        ),
        pytest.param(
            [5, 5, 5, 5],
            pytest.raises(ValueError, match="Depth should be increasing"),
            id="stable depth profile line",
        ),
        pytest.param(
            [1, 4, 3, 9],
            pytest.raises(ValueError, match="Depth should be increasing"),
            id="unstable depth profile line"),
    ],
)


def test_read_prof(d_prof: tuple[float, float, float, float],
                   expected: list[float],
) -> None:
    with expected as e:
        assert check_soundspeed_profile(d_prof) == e


@pytest.mark.parametrize(
    ("line", "z_max", "expected"),
    [
        pytest.param(
            "250 1600.0 0.0 1.75 1.05 0.0 /",
            250,
            nullcontext([250, 1600.0, 0.0, 1.75, 1.05, 0.0]),
            id="valid entry",
        ),
        pytest.param(
            "250 1600.0 0.0 /", 250,
            pytest.raises(ValueError, match="Bottom properties line: wrong number of element"),
            id="too few parameters",
        ),
        pytest.param(
                "250 1600.0 0.0 1.75 1.05 0.0 abc 789 /", 250,
                pytest.raises(
                    ValueError,
                    match="Bottom properties line: wrong number of element",
                ),
                id="too many parameters",
        ),
        pytest.param(
            "100 1600.0 0.0 1.75 1.05 0.0 /", 250,
            pytest.raises(ValueError, match="Bottom properties line: wrong z_max"),
            id="Depth must be z_max",
        ),
    ],
)
def test_read_bot_prop(line: str, z_max: float, expected: list[float]) -> None:
    with expected as e:
        assert read_bottom_properties(line, z_max) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            "E",
            nullcontext(),
            id="valid entry"
        ),
        pytest.param(
            "M",
            pytest.raises(ValueError, match="Incorrect run type"),
            id="unknown run type 1",
        ),
        pytest.param(
            "EA",
            pytest.raises(ValueError, match="Incorrect run type"),
            id="unknown run type 2",
        ),
        pytest.param(
            "5",
            pytest.raises(ValueError, match="Incorrect run type"),
            id="unknown run type 3"),
    ],
)
def test_read_run_type(line: str, expected: str) -> None:
    with expected as e:
        assert check_run_type(line) == e


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
