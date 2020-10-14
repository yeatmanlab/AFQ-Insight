import numpy as np
import pytest

from afqinsight.utils import ecdf


@pytest.mark.parametrize("reverse", [True, False])
def test_ecdf(reverse):
    n_pts = 20
    x_in = np.arange(n_pts)
    x, y = ecdf(x_in[::-1], reverse=reverse)
    if reverse:
        assert np.allclose(x, np.flip(x_in))  # nosec
    else:
        assert np.allclose(x, x_in)  # nosec
    assert np.allclose(y, np.linspace(1, 0, n_pts, endpoint=False)[::-1])  # nosec
