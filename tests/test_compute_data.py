"""Tests for functions from compute_data"""

import numpy.testing as npt
import math
from unittest.mock import Mock

def test_compute_data_mock_source():
    from inflammation.compute_data import analyse_data
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [[[0, 2, 0]],
                                                       [[0, 1, 0]]]

    result = analyse_data(data_source)
    npt.assert_array_almost_equal(result, [0, math.sqrt(0.25), 0])

# def test_analyse_data():
#     from inflammation.compute_data import analyse_data
#     path = Path.cwd() / "../data"
#     data_source = CSVDataSource(path)
#     result = analyse_data(data_source)
#
#     npt.assert_array_almost_equal(result, [0, math.sqrt(0.25), 0])
#
#     # TODO: add assert statement(s) to test the result value is as expected