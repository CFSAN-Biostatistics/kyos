# -*- coding: utf-8 -*-

"""
test_nn
----------------------------------

Tests for `nn` module.
"""

from __future__ import absolute_import

import numpy as np

from kyos import features
from kyos import nn


def test_load_train_data_all_classes(tmpdir):
    """Verify training features and targets are loaded properly when all
    output classes are represented in the data file.
    """

    # Header row. Names don't matter for this test
    columns = features.feature_names + [features.target_label_name]
    content = '\t'.join(columns) + '\n'

    # Feature data rows, one row for each possible target class
    for target in nn.output_classes:
        row = [str(i) for i, col in enumerate(features.feature_names)] + [target]
        content += '\t'.join(row)
        content += '\n'

    path = tmpdir.join("ftrFile.tsv")
    path.write(content)

    ftrs, targets = nn.load_train_data(str(path), 3, 5, scaling=None)

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


def test_load_train_data_some_classes(tmpdir):
    """Verify training features and targets are loaded properly when some
    output classes are not represented in the data file.
    """

    # Header row. Names don't matter for this test
    columns = features.feature_names + [features.target_label_name]
    content = '\t'.join(columns) + '\n'

    # Feature data rows, one row for each possible target class
    row = [str(i) for i, col in enumerate(features.feature_names)] + [nn.output_classes[0]]
    content += '\t'.join(row)
    content += '\n'
    row = [str(i) for i, col in enumerate(features.feature_names)] + [nn.output_classes[-2]]
    content += '\t'.join(row)
    content += '\n'

    path = tmpdir.join("ftrFile.tsv")
    path.write(content)

    ftrs, targets = nn.load_train_data(str(path), 3, 5, scaling=None)

    assert(ftrs.dtype == np.float32)
    assert(targets.dtype == np.float32)

    assert(np.array_equal(ftrs, np.array(
      [[3., 4., 5.],
       [3., 4., 5.]])))

    assert(np.array_equal(targets, np.array(
      [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0.]])))


def test_load_test_data_all_classes(tmpdir):
    """Verify training features and targets are loaded properly when all
    output classes are represented in the data file.
    """

    # Header row. Names don't matter for this test
    columns = features.feature_names + [features.target_label_name]
    content = '\t'.join(columns) + '\n'

    # Feature data rows, one row for each possible target class
    refs = ['A', 'G', 'G', 'C', 'C', 'C', 'T', 'G', 'A']
    for idx, target in enumerate(nn.output_classes):
        row = [str(i) for i, col in enumerate(features.feature_names[0:-1])] + [refs[idx]] + [target]
        content += '\t'.join(row)
        content += '\n'

    path = tmpdir.join("ftrFile.tsv")
    path.write(content)

    ftrs, targets, refs = nn.load_test_data(str(path), 3, 5, scaling=None)

    assert(ftrs.dtype == np.float32)
    assert(targets.dtype == np.int64)
    assert(refs.dtype == np.int64)

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
      [0, 1, 2, 3, 4, 5, 6, 7, 8])))

    assert(np.array_equal(refs, np.array(
      [0, 3, 3, 2, 2, 2, 1, 3, 0])))
