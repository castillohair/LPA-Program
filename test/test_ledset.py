"""
Unit tests for the LEDSet class

"""

import unittest

import numpy
import pandas

import lpadesign

class TestLEDSet(unittest.TestCase):
    """
    Tests for the LEDSet class.

    """
    def setUp(self):
        self.file_name = "test/test_lpa_files/led-archives/EO_10/Tiffani_c1/"+ \
            "EO_10_Tiffani_c1.xlsx"

    def test_create_ledset(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Check attributes
        self.assertEqual(led_set.name, 'TestLEDSet')
        self.assertEqual(led_set.lpa_name, "Tiffani")
        self.assertEqual(led_set.n_rows, 4)
        self.assertEqual(led_set.n_cols, 6)
        self.assertEqual(led_set.channel, 0)

    def test_get_intensity_1(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get intensities
        intensity = led_set.get_intensity(gs=1000., dc=8, gcal=215)
        # Test
        self.assertEqual(len(intensity), 24)
        numpy.testing.assert_almost_equal(
            intensity,
            numpy.array([11.3110594560676,
                         11.5279113045840,
                         11.2051557471600,
                         11.5112113864180,
                         11.2864295928559,
                         11.0325353675551,
                         11.4508611229791,
                         11.5422326753419,
                         11.0743943419891,
                         11.3080330670466,
                         11.3920577957224,
                         11.1096415717709,
                         11.2951734054688,
                         11.4739246978975,
                         10.2214698525038,
                         11.0400127776493,
                         11.4054846036568,
                         11.4645629503993,
                         11.2170686811425,
                         11.3613838462866,
                         11.1145737192158,
                         11.2471841967786,
                         11.8171084339057,
                         10.6232870495689,
                         ]))

    def test_get_intensity_2(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get intensities
        gs = numpy.array([4075,
                          1523,
                          945,
                          3750,
                          2390,
                          3491,
                          1081,
                          2115,
                          2498,
                          2570,
                          3031,
                          3103,
                          1905,
                          50,
                          3271,
                          3212,
                          3694,
                          3865,
                          1550,
                          405,
                          1970,
                          3879,
                          3830,
                          2303,
                          ])
        intensity = led_set.get_intensity(gs=gs, dc=8, gcal=215)
        # Test
        self.assertEqual(len(intensity), 24)
        numpy.testing.assert_almost_equal(
            intensity,
            numpy.array([46.092567283476,
                         17.557008916882,
                         10.588872181066,
                         43.167042699068,
                         26.974566726926,
                         38.514580968135,
                         12.378380873940,
                         24.411822108348,
                         27.663837066289,
                         29.061644982310,
                         34.529327178834,
                         34.473217797205,
                         21.517305337418,
                         0.573696234895,
                         33.434427887540,
                         35.460521041809,
                         42.131860125908,
                         44.310535803293,
                         17.386456455771,
                         4.601360457746,
                         21.895710226855,
                         43.627827499304,
                         45.259525301859,
                         24.465430075157,
                         ]))

    def test_get_intensity_3(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get intensities
        intensity = led_set.get_intensity(gs=50, dc=8, gcal=215, row=2, col=1)
        # Test
        self.assertEqual(len(intensity), 1)
        numpy.testing.assert_almost_equal(
            intensity,
            0.573696234895)

    def test_get_intensity_4(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get intensities
        gs = numpy.array([2390, 3491, 1550])
        row = [0, 0, 3]
        col = [4, 5, 0]
        intensity = led_set.get_intensity(
            gs=gs,
            dc=8,
            gcal=215,
            row=row,
            col=col)
        # Test
        self.assertEqual(len(intensity), 3)
        numpy.testing.assert_almost_equal(
            intensity,
            numpy.array([26.974566726926,
                         38.514580968135,
                         17.386456455771,
                         ]))

    def test_get_intensity_5(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get intensities
        gs = numpy.array([2390, 3491, 1550, 1905])
        dc = numpy.array([8, 7, 4, 8])
        gcal = numpy.array([215, 215, 125, 100])
        row = [0, 0, 3, 2]
        col = [4, 5, 0, 0]
        intensity = led_set.get_intensity(
            gs=gs,
            dc=dc,
            gcal=gcal,
            row=row,
            col=col)
        # Test
        self.assertEqual(len(intensity), 4)
        numpy.testing.assert_almost_equal(
            intensity,
            numpy.array([26.974566726926,
                         33.700258347118,
                         5.0542024580729,
                         10.0080489941479,
                         ]))

    def test_get_grayscale_1(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        grayscale = led_set.get_grayscale(intensity=20., dc=6, gcal=199)
        # Test
        self.assertEqual(len(grayscale), 24)
        numpy.testing.assert_array_equal(
            grayscale,
            numpy.array([2547,
                         2499,
                         2571,
                         2503,
                         2553,
                         2611,
                         2516,
                         2496,
                         2602,
                         2548,
                         2529,
                         2593,
                         2551,
                         2511,
                         2819,
                         2610,
                         2526,
                         2513,
                         2568,
                         2536,
                         2592,
                         2562,
                         2438,
                         2712,
                         ]))

    def test_get_grayscale_2(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        intensity = numpy.array([14.7705276,
                                 18.011532,
                                 14.49721108,
                                 7.928343284,
                                 12.41515464,
                                 16.13671401,
                                 20.16590141,
                                 16.30123003,
                                 21.19584201,
                                 12.63006767,
                                 14.17082388,
                                 21.96112159,
                                 22.46311068,
                                 21.2878999,
                                 8.064405988,
                                 8.002552636,
                                 21.14038489,
                                 13.07107565,
                                 17.66874661,
                                 9.344051306,
                                 14.95665436,
                                 7.403629286,
                                 8.370799823,
                                 9.238055192,
            ])
        grayscale = led_set.get_grayscale(intensity=intensity, dc=6, gcal=199)
        # Test
        self.assertEqual(len(grayscale), 24)
        numpy.testing.assert_array_equal(
            grayscale,
            numpy.array([1881,
                         2251,
                         1864,
                         992,
                         1585,
                         2107,
                         2537,
                         2034,
                         2757,
                         1609,
                         1792,
                         2848,
                         2865,
                         2673,
                         1137,
                         1044,
                         2670,
                         1642,
                         2269,
                         1185,
                         1938,
                         948,
                         1020,
                         1253,
                         ]))

    def test_get_grayscale_3(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        grayscale = led_set.get_grayscale(intensity=7.92834328438,
                                          dc=6,
                                          gcal=199,
                                          row=0,
                                          col=3)
        # Test
        self.assertEqual(len(grayscale), 1)
        numpy.testing.assert_array_equal(
            grayscale,
            992)

    def test_get_grayscale_4(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        intensity = numpy.array([7.92834328438,
                                 21.961121594,
                                 9.23805519165,
                                 ])
        row = [0, 1, 3]
        col = [3, 5, 5]
        grayscale = led_set.get_grayscale(intensity=intensity,
                                          dc=6,
                                          gcal=199,
                                          row=row,
                                          col=col)
        # Test
        self.assertEqual(len(grayscale), 3)
        numpy.testing.assert_array_equal(
            grayscale,
            numpy.array([992,
                         2848,
                         1253,
                ]))

    def test_get_grayscale_5(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        intensity = numpy.array([7.92834328438,
                                 21.961121594,
                                 9.23805519165,
                                 21.195842008,
                                 ])
        row = [0, 1, 3, 1]
        col = [3, 5, 5, 2]
        grayscale = led_set.get_grayscale(intensity=intensity,
                                          dc=6,
                                          gcal=199,
                                          row=row,
                                          col=col)
        # Test
        self.assertEqual(len(grayscale), 4)
        numpy.testing.assert_array_equal(
            grayscale,
            numpy.array([992,
                         2848,
                         1253,
                         2757,
                ]))

    def test_get_grayscale_6(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        intensity = numpy.array([7.92834328438,
                                 21.961121594,
                                 9.23805519165,
                                 21.195842008,
                                 ])
        dc = numpy.array([6, 4, 3, 8])
        gcal = numpy.array([199, 230, 150, 255])
        row = [0, 1, 3, 1]
        col = [3, 5, 5, 2]
        grayscale = led_set.get_grayscale(intensity=intensity,
                                          dc=dc,
                                          gcal=gcal,
                                          row=row,
                                          col=col)
        # Test
        self.assertEqual(len(grayscale), 4)
        numpy.testing.assert_array_equal(
            grayscale,
            numpy.array([992,
                         3696,
                         3324,
                         1614,
                ]))

    def test_get_grayscale_error(self):
        # Load
        led_set = lpadesign.LEDSet(name='TestLEDSet', file_name=self.file_name)
        # Get grayscale values
        intensity = numpy.array([7.92834328438,
                                 21.961121594,
                                 9.23805519165,
                                 21.195842008,
                                 ])
        dc = numpy.array([6, 4, 2, 8])
        gcal = numpy.array([199, 230, 150, 255])
        row = [0, 1, 3, 1]
        col = [3, 5, 5, 2]
        with self.assertRaises(ValueError):
            grayscale = led_set.get_grayscale(intensity=intensity,
                                              dc=dc,
                                              gcal=gcal,
                                              row=row,
                                              col=col)
