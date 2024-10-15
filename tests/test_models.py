"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize('test, expected, expected_raises', [
    (
        [0, 0, 0],
        0.0,
        None,
    ),
    (
        [1.0, 1.0, 1.0],
        0,
        None,
    ),
    (
        [0.0, 2.0],
        1.0,
        None
    ),
    (
        ["4", "5"],
        None,
        TypeError
    ),
    (
        None,
        None,
        ValueError
    ),
])
def test_daily_standard_deviation(test, expected, expected_raises):
    from inflammation.models import std_dev
    if expected_raises:
        with pytest.raises(expected_raises):
            result_data = std_dev(test)['standard deviation']
            npt.assert_approx_equal(result_data, expected)
    else:
        result_data = std_dev(test)['standard deviation']
        npt.assert_approx_equal(result_data, expected)
