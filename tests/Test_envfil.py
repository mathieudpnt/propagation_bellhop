from pathlib import Path

from utils import utils_acoustic_toolbox

#@pytest.fixture


def test_file_ray (filename:Path) -> None: #ok
    assert filename.exists()
    assert filename.suffix == ".env"
    assert filename.stat().st_size > 0
    try: #voir si fichier corrompu
        _ = filename.read_text()
    except Exception:
        assert False

def test_file_env(_,filename=utils_acoustic_toolbox.write_env_file())-> None: #ok
    fid = Path.open(filename)

    next(fid), next(fid)
    assert (fid.readline().split()[0])==1
    next(fid), next(fid)
    data = (fid.readline()).split()
    zmin = float(data[1])
    zmax = float(data[2])
    assert zmin >= 0
    assert zmax > 0
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
    next(fid), next(fid), next(fid)
    assert round((float(fid.readline().split()[0])*1000),0)==zmax
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

#@pytest.mark.parametrize("value, min, max", [(freq, 0, 120000),(soundspeed, 1300, 1600)]
