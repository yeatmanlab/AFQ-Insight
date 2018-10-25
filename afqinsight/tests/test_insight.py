from __future__ import absolute_import, division, print_function

import afqinsight as afqi
import numpy as np
import os.path as op

data_path = op.join(afqi.__path__[0], 'data')
test_data_path = op.join(data_path, 'test_data')


def test__sigmoid():
    z = np.array([-np.inf, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, np.inf])

    sig_z = np.array([
        0.00000000e+00, 4.53978687e-05, 2.68941421e-01, 3.77540669e-01,
        5.00000000e-01, 6.22459331e-01, 7.31058579e-01, 9.99954602e-01,
        1.00000000e+00
    ])

    assert np.allclose(afqi.insight._sigmoid(z), sig_z)
