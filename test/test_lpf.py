"""
Unit tests for the LPF class

"""

import filecmp
import os
import shutil
import unittest

import numpy
import pandas

import lpaprogram

class TestLPF(unittest.TestCase):
    """
    Tests for the LPF class.

    """
    def setUp(self):
        # The following file was obtained from Iris
        self.file_name = "test/test_lpf_files/program.lpf"
        # Expected contents extracted from file using a hex editor
        self.file_version_expected = 1
        self.n_channels_expected = 48
        self.step_size_expected = 1000
        self.n_steps_expected = 61
        self.gs_expected = numpy.array([[3353,  284,
                                          828, 2066,
                                         1274, 1823,
                                         1691, 3073,
                                         3407, 1705,
                                         1152, 2023,
                                          889, 3382,
                                         1431, 3673,
                                         1037, 3383,
                                         3496, 2867,
                                         3715, 1737,
                                         3549,  666,
                                          140, 2023,
                                          820,  166,
                                         1808,  563,
                                         3408,  730,
                                         1209, 3272,
                                         3213, 3502,
                                          979, 1319,
                                         2629,  650,
                                         1070, 1090,
                                         2984,  408,
                                          922,  814,
                                         3764, 3640,
                                         ]])
        self.gs_expected = numpy.repeat(self.gs_expected, 61, axis=0)
        # Directory where to save temporary files
        self.temp_dir = "test/temp_lpf"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create(self):
        lpf = lpaprogram.LPF()

    def test_create_and_load(self):
        lpf = lpaprogram.LPF(self.file_name)
        # Check header info
        self.assertEqual(lpf.file_version, self.file_version_expected)
        self.assertEqual(lpf.n_channels, self.n_channels_expected)
        self.assertEqual(lpf.step_size, self.step_size_expected)
        self.assertEqual(lpf.n_steps, self.n_steps_expected)
        # Check size of grayscale array
        self.assertEqual(lpf.grayscale.shape, self.gs_expected.shape)
        # Check contents of grayscale array
        numpy.testing.assert_array_equal(lpf.grayscale, self.gs_expected)

    def test_create_then_load(self):
        lpf = lpaprogram.LPF()
        lpf.load(self.file_name)
        # Check header info
        self.assertEqual(lpf.file_version, self.file_version_expected)
        self.assertEqual(lpf.n_channels, self.n_channels_expected)
        self.assertEqual(lpf.step_size, self.step_size_expected)
        self.assertEqual(lpf.n_steps, self.n_steps_expected)
        # Check size of grayscale array
        self.assertEqual(lpf.grayscale.shape, self.gs_expected.shape)
        # Check contents of grayscale array
        numpy.testing.assert_array_equal(lpf.grayscale, self.gs_expected)

    def test_save(self):
        # Create LPF object
        lpf = lpaprogram.LPF()
        lpf.file_version = self.file_version_expected
        lpf.n_channels = self.n_channels_expected
        lpf.step_size = self.step_size_expected
        lpf.n_steps = self.n_steps_expected
        lpf.grayscale = self.gs_expected
        # Attempt to save
        lpf.save(os.path.join(self.temp_dir, 'program.lpf'))
        # Check if file is identical with source
        comp = filecmp.cmp(self.file_name, os.path.join(self.temp_dir,
                                                        'program.lpf'))
        self.assertTrue(comp)
