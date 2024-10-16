"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: a 2D data array containing inflammation data
    :returns: an array with the mean values for each day"""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: a 2D data array containing inflammation data
    :returns: an array with the maximum values for each day"""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: a 2D data array containing inflammation data
    :returns: an array with the minimum values for each day"""
    return np.min(data, axis=0)

def std_dev(data):
    """Computes and returns standard deviation for a 2D inflammation data array."""
    return np.std(data, axis=0)

def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array

    NaN values are ignored, and normalised to 0

    """
    if not isinstance(data, np.ndarray):
        raise TypeError('Input should be an ndarray')
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')

    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised

