from contextlib import nullcontext
from pathlib import Path

import pytest
from core_utils import check_empty_file, check_file_exist, check_suffix
from numpy import ndarray
from reader_utils import (
    check_media,
    check_run_type,
    check_soundspeed_profile,
    read_angle,
    read_bottom_properties,
    read_coord_type,
    read_depth,
    read_dim,
    read_env_param,
    read_r,
    read_z,
)


def test_invalid_path() -> None:
    invalid_path = Path(r"wrong_path\that_does\not_exist")
    with pytest.raises(FileNotFoundError):
        check_file_exist(invalid_path)


@pytest.mark.parametrize(
    ("filename", "suffix", "expected"),
    [
        pytest.param(
            "file.expected_suffix",
            ".expected_suffix",
            nullcontext(),
            id="valid entry",
        ),
        pytest.param(
            "file.wrong_suffix",
            ".right_suffix",
            pytest.raises(
                ValueError,
                match=r"not a .right_suffix file"
            ),
            id="wrong suffix",
        ),
    ],
)
def test_invalid_suffix(
        tmp_path: Path,
        filename: str,
        suffix: str,
        expected: list[str]
) -> None:
    path = tmp_path / filename
    path.touch()
    path.write_text("test")
    with expected as e:
        assert check_suffix(path, suffix) == e


def test_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_file.txt"
    empty_file.touch()
    with pytest.raises(ValueError, match="is empty"):
        check_empty_file(empty_file)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            "1 2 3 4",
            nullcontext([1, 2, 3, 4]),
            id="valid entry",
        ),
        pytest.param(
            "1 2 3 4 /",
            nullcontext([1, 2, 3, 4]),
            id="another valid entry",
        ),
        pytest.param(
            "1 2 3 4 5",
            pytest.raises(
                ValueError,
                match="Invalid environmental characteristics line"
            ),
            id="too many values",
        ),
        pytest.param(
            "1 2",
            pytest.raises(ValueError,
                          match="Invalid environmental characteristics line"),
            id="not enough values",
        ),
    ],
)
def test_read_environmental_line(line: str, expected: str) -> None:
    with expected as e:
        assert read_env_param(line) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            1,
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
def test_read_media_line(line: int, expected: int) -> None:
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
def test_read_depth_line(line: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_depth(line) == e


@pytest.mark.parametrize(
    ("d_prof", "expected"),
    [
        pytest.param(
            [1, 2, 3, 4],
            nullcontext(),
            id="valid depth profile"
        ),
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
def test_check_soundspeed_profile(
        d_prof: list[float],
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
            "250 1600.0 0.0 /",
            250,
            pytest.raises(
                ValueError,
                match="Bottom properties line: wrong number of element"
            ),
            id="too few parameters",
        ),
        pytest.param(
            "250 1600.0 0.0 1.75 1.05 0.0 abc 789 /",
            250,
            pytest.raises(
                ValueError,
                match="Bottom properties line: wrong number of element",
            ),
            id="too many parameters",
        ),
        pytest.param(
            "100 1600.0 0.0 1.75 1.05 0.0 /", 250,
            pytest.raises(
                ValueError,
                match="Bottom properties line: wrong z_max"
            ),
            id="depth != z_max",
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
        pytest.param(
            "-120 120",
            nullcontext((-120.0, 120.0)),
            id="valid angles",
        ),
        pytest.param(
            "-120 1220",
            pytest.raises(ValueError, match="Invalid angle line: -120 1220"),
            id="z_max > 180",
        ),
        pytest.param(
            "-190 120",
            pytest.raises(ValueError, match="Invalid angle line: -190 120"),
            id="z_min < 180",
        ),
        pytest.param(
            "-190 190",
            pytest.raises(ValueError, match="Invalid angle line: -190 190"),
            id="z_min < 180 and z_max > 180",
        ),
        pytest.param(
            "120 -120",
            pytest.raises(ValueError, match="Invalid angle line: 120 -120"),
            id="z_min > z_max",
        ),
    ],
)
def test_read_angle(line: str, expected: tuple[float, float]) -> None:
    with expected as e:
        assert read_angle(line) == e


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            "'rz'",
            nullcontext("'rz'"),
            id="valid depth",
        ),
        pytest.param(
            "'r'",
            pytest.raises(ValueError, match="Invalid coordinate type"),
            id="coordinate type must be 'rz'",
        ),
        pytest.param(
            "250",
            pytest.raises(ValueError, match="Invalid coordinate type"),
            id="coordinate type must be str",
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
