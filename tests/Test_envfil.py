from pathlib import Path

import pytest


@pytest.fixture
def test_file_env(root : Path): #ok
    assert filename.exists()
    assert filename.suffix == ".env"
    assert filename.stat().st_size > 0

    fid = Path.open(filename)

    next(fid), next(fid)
    next(fid), next(fid), next(fid)
    data = (fid.readline()).split()
    zmin=float(data[1])
    zmax=float(data[2])
    assert zmin < zmax
    z0=float(fid.readline().split()[0])
    assert z0==zmin
    depth= [z0]

    while True:
        line = fid.readline()
        parts = line.split()
        if not parts:
            continue
        try:
            z = float(parts[0])
            if z > zmax:
                break
            depth.append(z)
        except ValueError:
            break

    assert depth[-1]==zmax
    bot_prop=fid.readline().split()
    assert len(bot_prop)==7  # noqa: PLR2004
    next(fid), next(fid)
    next(fid), next(fid), next(fid), next(fid)
    run_type=fid.readline().split()[0]
    assert run_type in {"E", "I", "A", "R"}
    next(fid)
    ang=fid.readline().split()
    assert len(ang)==3  # noqa: PLR2004
    assert -180<float(ang[0])<180  # noqa: PLR2004
    assert -180<float(ang[1])<180  # noqa: PLR2004
    assert float(ang[0])<float(ang[1])
    info=fid.readline().split()
    assert len(info)==3  # noqa: PLR2004
