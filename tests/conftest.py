from pathlib import Path

import pytest

ENV_FILE_VALID = """'valid'
14000
1
'CVWF'
19.3 35. 8. 50.
0 0.0 250
0.0 1504.7250415797987 /
2.096774193548387 1504.8726490874922 /
62.90322580645161 1522.5374940594131 /
65.0 1522.5572327650782 /
250 1522.5572327650782 /
A* 0.0
250 1600.0 0.0 1.75 1.05 0.0 /
1
5 /
1
20 /
1
3.021188326442315 /
A
50001
-20 20 /
0. 260 3.17
"""

ENV_FILE_INVALID1 = """'bad_angle_inf'
14000
1
'CVWF'
19.3 35. 8. 50.
0 0.0 250
0.0 1504.7250415797987 /
2.096774193548387 1504.8726490874922 /
62.90322580645161 1522.5374940594131 /
65.0 1522.5572327650782 /
250 1522.5572327650782 /
A* 0.0
250 1600.0 0.0 1.75 1.05 0.0 /
1
5 /
1
20 /
1
3.021188326442315 /
A
50001
-183 20 /
0. 260 3.17
"""


@pytest.fixture(
    params=[ENV_FILE_VALID, ENV_FILE_INVALID1],
    ids=lambda c: c.splitlines()[0].strip().strip("'\""),
)
def sample_env(tmp_path: Path, request) -> Path:
    content = request.param
    name = content.splitlines()[0].strip().strip("'\"")
    env_path = tmp_path / f"sample_{name}.env"
    env_path.write_text(content)
    return env_path
