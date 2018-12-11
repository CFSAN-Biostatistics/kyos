# -*- coding: utf-8 -*-

"""
test_nn
----------------------------------

Tests for `nn` module.
"""

from __future__ import absolute_import

import numpy as np

from kyos import nn


def test_load_train_data(tmpdir):
    """Verify training features and targets are loaded properly."""

    # Header row. Names don't matter for this test
    content = '\t'.join(['C' + str(i) for i in range(0, 21)]) + '\n'

    # Feature data rows, one row for each possible target class
    for target in nn.output_classes:
        row = [str(i) for i in range(0, 20)] + [target]
        content += '\t'.join(row)
        content += '\n'

    path = tmpdir.join("ftrFile.tsv")
    path.write(content)

    ftrs, targets = nn.load_train_data(str(path), 3, 5)

    assert(ftrs.dtype == np.float32)
    assert(targets.dtype == np.float32)

    assert(np.array_equal(ftrs, np.array(
      [[3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.],
       [3., 4., 5.]])))

    assert(np.array_equal(targets, np.array(
      [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1.]])))
