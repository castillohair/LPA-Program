"""
Unit tests for the LPA class

"""

import filecmp
import os
import shutil
import unittest

import numpy
import pandas

import lpadesign

class TestLPA(unittest.TestCase):
    """
    Tests for the LPA class.

    """
    def setUp(self):
        lpadesign.LED_DATA_PATH = "test/test_lpa_files/led-archives"
        # Contents of LPA files in test_lpa_files/Jennie
        self.folder_to_load = 'test/test_lpa_files/Jennie'
        self.dc_file_to_load = os.path.join(self.folder_to_load, 'dc.txt')
        self.gcal_file_to_load = os.path.join(self.folder_to_load, 'gcal.txt')
        self.lpf_file_to_load = os.path.join(self.folder_to_load, 'program.lpf')
        self.dc_to_load_exp = numpy.array([[[9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7]],

                                           [[9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7]],

                                           [[9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7]],

                                           [[9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 7],
                                            [9, 8]]], dtype=int)
        self.gcal_to_load_exp = numpy.array([[[235, 182],
                                              [231, 182],
                                              [210, 192],
                                              [201, 153],
                                              [208, 178],
                                              [218, 171]],

                                             [[210, 197],
                                              [236, 174],
                                              [208, 189],
                                              [205, 177],
                                              [201, 196],
                                              [205, 186]],

                                             [[209, 175],
                                              [203, 175],
                                              [204, 187],
                                              [226, 182],
                                              [202, 183],
                                              [223, 231]],

                                             [[242, 182],
                                              [210, 199],
                                              [201, 215],
                                              [207, 231],
                                              [218, 177],
                                              [207, 222]]], dtype=int)
        self.intensity_to_load_exp = numpy.array([0.0000000000000000,
                                                 22.9901431948947,
                                                  0.0000000000000000,
                                                  0.0000000000000000,
                                                  0.0242301454276248,
                                                  0.0000000000000,
                                                  0.0362731297390563,
                                                  0.0000000000000,
                                                  0.0482746603603435,
                                                  0.0000000000000,
                                                  0.0604583786359338,
                                                  0.0000000000000,
                                                  0.0847081438994608,
                                                  0.0000000000000,
                                                  0.1209327131758000,
                                                  0.0000000000000,
                                                  0.1811383239257640,
                                                  0.0000000000000,
                                                  0.2544416023194100,
                                                  0.0000000000000,
                                                  0.3752463358481230,
                                                  0.0000000000000,
                                                  0.5317364340341630,
                                                  0.0000000000000,
                                                  0.7718052799126070,
                                                  0.0000000000000,
                                                  1.1380987428642800,
                                                  0.0000000000000,
                                                  1.6598674265846700,
                                                  0.0000000000000,
                                                  2.4072649585467100,
                                                  0.0000000000000,
                                                  3.5174299255496300,
                                                  0.0000000000000,
                                                  5.1212377228010200,
                                                  0.0000000000000,
                                                  7.4885242535130900,
                                                  0.0000000000000,
                                                 10.9275358027087000,
                                                  0.0000000000000,
                                                 15.9178976564621000,
                                                  0.0000000000000,
                                                 23.2680373617547000,
                                                  0.0000000000000,
                                                 33.9844061329441000,
                                                  0.0000000000000,
                                                 49.3810944000507000,
                                                  0.0000000000000])
        self.intensity_to_load_exp.resize(1,4,6,2)
        self.intensity_to_load_exp = numpy.repeat(self.intensity_to_load_exp,
                                                  61,
                                                  axis=0)
        self.step_size_to_load_exp = 1000
        # Directory where to save temporary files
        self.temp_dir = "test/temp_lpa"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        # Test data to save
        self.dc_to_save = numpy.array([[[ 49,  35],
                                        [ 60,  52],
                                        [ 29,  35],
                                        [ 10,  31],
                                        [ 45,  55],
                                        [  7,  58]],

                                       [[ 39,  41],
                                        [ 32,  12],
                                        [ 62,  44],
                                        [  1,  24],
                                        [ 41,  23],
                                        [ 30,  26]],

                                       [[ 23,   4],
                                        [ 44,  62],
                                        [ 42,  44],
                                        [ 37,  33],
                                        [ 58,  32],
                                        [ 19,  12]],

                                       [[ 53,   1],
                                        [ 15,  44],
                                        [ 18,  51],
                                        [ 48,  10],
                                        [ 29,  39],
                                        [ 20,  18]]], dtype=int)
        self.gcal_to_save = numpy.array([[[168, 100],
                                          [126, 240],
                                          [ 57,  24],
                                          [160, 250],
                                          [158, 185],
                                          [ 71,  38]],

                                         [[219, 240],
                                          [187,  65],
                                          [154, 205],
                                          [236, 243],
                                          [159,  88],
                                          [ 64, 237]],

                                         [[113, 143],
                                          [197, 164],
                                          [ 57, 252],
                                          [144,  21],
                                          [234, 118],
                                          [105, 238]],

                                         [[181, 152],
                                          [ 84, 230],
                                          [ 36, 153],
                                          [176, 137],
                                          [215, 177],
                                          [208,  87]]], dtype=int)
        self.intensity_to_save = numpy.array(
            [[[[  7.26141288e+01,   2.47857613e+01],
               [  1.18674612e+02,   1.61206558e+02],
               [  2.22407543e+01,   1.15681545e+01],
               [  1.14427120e+00,   1.01372057e+02],
               [  1.18811366e+02,   8.10349636e+01],
               [  7.78302710e+00,   1.56174145e+01]],

              [[  1.31245761e+02,   5.89023604e+01],
               [  2.35087729e+01,   3.44806540e+00],
               [  2.01714425e+02,   7.30213626e+01],
               [  1.97138556e+00,   4.06834317e+01],
               [  1.72172707e+02,   8.78222159e-01],
               [  1.92415490e+00,   1.78975793e+01]],

              [[  6.57509960e+01,   1.69377638e+00],
               [  8.83463844e+01,   1.55293150e+02],
               [  3.25440445e+00,   9.05380744e+01],
               [  6.92365671e+01,   8.61622996e+00],
               [  2.01227485e+01,   6.02829722e+00],
               [  3.32210814e+01,   2.21158389e+01]],

              [[  1.30707259e+02,   5.36804858e-01],
               [  2.27828432e+01,   9.07896749e+01],
               [  1.62059506e+01,   4.83733422e+00],
               [  1.36676940e+02,   4.40683291e-01],
               [  8.17877776e+01,   1.00947668e+02],
               [  1.64523709e+01,   6.07867565e+00]]],


             [[[  9.51268587e+01,   2.05134179e+00],
               [  1.50984392e+01,   9.26841392e+01],
               [  2.66168532e+01,   1.37749592e+01],
               [  3.88838326e+01,   1.00725603e+02],
               [  1.58140128e+02,   3.00881528e+01],
               [  7.09079407e+00,   6.91401468e+00]],

              [[  5.34279618e+01,   1.27286956e+02],
               [  1.29604887e+02,   1.43852410e+01],
               [  2.23025323e+02,   2.62613905e+01],
               [  6.24582060e+00,   2.36320916e+01],
               [  3.06220522e+01,   3.96013140e+00],
               [  3.04594978e+01,   2.72917839e+01]],

              [[  3.68412195e+01,   8.12697052e-01],
               [  3.88310506e+01,   8.49883818e+01],
               [  4.74732299e+01,   1.58734318e+02],
               [  7.98200973e+01,   3.83711116e+00],
               [  1.41761605e+02,   1.36377871e+01],
               [  4.22137052e+01,   3.90868457e+01]],

              [[  1.79622573e+02,   1.90742843e-02],
               [  3.06568003e+01,   6.10707525e+01],
               [  1.61670148e+01,   1.30258337e+01],
               [  1.09846340e+02,   1.95864563e+01],
               [  4.71097599e+01,   1.12435136e+02],
               [  8.07808717e-02,   3.97489157e+00]]],


             [[[  9.69128373e+01,   6.05839891e+01],
               [  4.63077492e+01,   2.00338638e+01],
               [  6.96149385e+00,   2.10138408e+00],
               [  3.67129256e+01,   4.74335573e+01],
               [  3.99246527e+01,   1.09600250e+02],
               [  1.19211104e+01,   1.22821700e+00]],

              [[  1.21183586e+02,   2.27011144e+01],
               [  4.02034086e+00,   1.38910915e+01],
               [  1.39814273e+01,   6.21145701e+01],
               [  2.53242453e+00,   7.45996129e-01],
               [  2.07200496e+01,   2.53708624e+00],
               [  4.67204932e+01,   1.10543872e+02]],

              [[  3.36086566e+01,   1.06965661e+01],
               [  1.56243281e+01,   1.73972937e+02],
               [  4.69202971e+00,   1.65661261e+02],
               [  9.28435431e+01,   3.55764889e+00],
               [  4.24111741e+00,   4.00733528e+01],
               [  3.67148779e+01,   3.01552586e+01]],

              [[  1.40191928e+02,   2.57639083e+00],
               [  2.71070656e+00,   1.34388314e+02],
               [  4.94052204e+00,   1.07907522e+02],
               [  6.43056497e+01,   7.85086862e+00],
               [  7.54371973e+00,   1.69532793e+01],
               [  6.24166868e+01,   7.75588600e-02]]]])
        self.step_size_to_save = 60000
        self.dc_to_save_exp = """49\t35\t60\t52\t29\t35\t10\t31\t45\t55\t7\t58
39\t41\t32\t12\t62\t44\t1\t24\t41\t23\t30\t26
23\t4\t44\t62\t42\t44\t37\t33\t58\t32\t19\t12
53\t1\t15\t44\t18\t51\t48\t10\t29\t39\t20\t18
"""
        self.gcal_to_save_exp = """168\t100\t126\t240\t57\t24\t160\t250\t158\t185\t71\t38
219\t240\t187\t65\t154\t205\t236\t243\t159\t88\t64\t237
113\t143\t197\t164\t57\t252\t144\t21\t234\t118\t105\t238
181\t152\t84\t230\t36\t153\t176\t137\t215\t177\t208\t87
"""
        self.file_version_to_save_exp = 1
        self.n_channels_to_save_exp = 48
        self.step_size_to_save_exp = 60000
        self.n_steps_to_save_exp = 3
        self.gs_to_save_exp = numpy.array(
          [[1545, 1607, 2696, 2929, 2099, 3292,  107, 2509, 2592, 1756, 2541,
            1475, 2400, 1466,  690,  942, 3275, 1888, 1272, 1527, 3947,  108,
             153,  663, 3946,  644, 1538, 3342,  206, 1856, 2185, 2744,  223,
             366, 2767, 2231, 2453,  788, 2824, 2224, 3746,  166, 2491,   92,
            2125, 3269,  611, 1254],
           [2024,  133,  343, 1684, 2512, 3920, 3636, 2493, 3450,  652, 2315,
             653,  977, 3168, 3804, 3930, 3621,  679, 4030,  887,  702,  487,
            2422, 1011, 2211,  309,  676, 1829, 3005, 3254, 2519, 1222, 1571,
             828, 3516, 3943, 3371,   28, 3800, 1496, 3737,  447, 2002, 4089,
            1224, 3641,    3,  820],
           [2062, 3928, 1052,  364,  657,  598, 3433, 1174,  871, 2375, 3892,
             116, 2216,  565,  118, 3795,  227, 1606, 1634,   28,  475,  312,
            3715, 4095, 2017, 4067,  272, 3744,  297, 3396, 2930, 1133,   47,
            2433, 3058, 3042, 2631, 3782,  336, 3292, 1142, 3703, 1172, 1639,
             196,  549, 2318,   16]])

        # Data for testing set_timecourse_staggered()
        self.timecourse_ch0 = numpy.linspace(0,20,231)
        self.timecourse_pre_ch0 = 2
        self.timecourse_sampling_steps_ch0 = numpy.arange(0,231,10).astype(int)
        self.intensity_well_ch0_1 = [
            numpy.append(2*numpy.ones(231), []),
            numpy.append(2*numpy.ones(221), numpy.linspace(0,20,231)[:10]),
            numpy.append(2*numpy.ones(211), numpy.linspace(0,20,231)[:20]),
            numpy.append(2*numpy.ones(201), numpy.linspace(0,20,231)[:30]),
            numpy.append(2*numpy.ones(191), numpy.linspace(0,20,231)[:40]),
            numpy.append(2*numpy.ones(181), numpy.linspace(0,20,231)[:50]),
            numpy.append(2*numpy.ones(171), numpy.linspace(0,20,231)[:60]),
            numpy.append(2*numpy.ones(161), numpy.linspace(0,20,231)[:70]),
            numpy.append(2*numpy.ones(151), numpy.linspace(0,20,231)[:80]),
            numpy.append(2*numpy.ones(141), numpy.linspace(0,20,231)[:90]),
            numpy.append(2*numpy.ones(131), numpy.linspace(0,20,231)[:100]),
            numpy.append(2*numpy.ones(121), numpy.linspace(0,20,231)[:110]),
            numpy.append(2*numpy.ones(111), numpy.linspace(0,20,231)[:120]),
            numpy.append(2*numpy.ones(101), numpy.linspace(0,20,231)[:130]),
            numpy.append(2*numpy.ones(91),  numpy.linspace(0,20,231)[:140]),
            numpy.append(2*numpy.ones(81),  numpy.linspace(0,20,231)[:150]),
            numpy.append(2*numpy.ones(71),  numpy.linspace(0,20,231)[:160]),
            numpy.append(2*numpy.ones(61),  numpy.linspace(0,20,231)[:170]),
            numpy.append(2*numpy.ones(51),  numpy.linspace(0,20,231)[:180]),
            numpy.append(2*numpy.ones(41),  numpy.linspace(0,20,231)[:190]),
            numpy.append(2*numpy.ones(31),  numpy.linspace(0,20,231)[:200]),
            numpy.append(2*numpy.ones(21),  numpy.linspace(0,20,231)[:210]),
            numpy.append(2*numpy.ones(11),  numpy.linspace(0,20,231)[:220]),
            numpy.append(2*numpy.ones(1),   numpy.linspace(0,20,231)[:230]),
            ]
        self.intensity_well_ch0_2 = [
            numpy.append(2*numpy.ones(720), []),
            numpy.append(2*numpy.ones(710), numpy.linspace(0,20,231)[:10]),
            numpy.append(2*numpy.ones(700), numpy.linspace(0,20,231)[:20]),
            numpy.append(2*numpy.ones(690), numpy.linspace(0,20,231)[:30]),
            numpy.append(2*numpy.ones(680), numpy.linspace(0,20,231)[:40]),
            numpy.append(2*numpy.ones(670), numpy.linspace(0,20,231)[:50]),
            numpy.append(2*numpy.ones(660), numpy.linspace(0,20,231)[:60]),
            numpy.append(2*numpy.ones(650), numpy.linspace(0,20,231)[:70]),
            numpy.append(2*numpy.ones(640), numpy.linspace(0,20,231)[:80]),
            numpy.append(2*numpy.ones(630), numpy.linspace(0,20,231)[:90]),
            numpy.append(2*numpy.ones(620), numpy.linspace(0,20,231)[:100]),
            numpy.append(2*numpy.ones(610), numpy.linspace(0,20,231)[:110]),
            numpy.append(2*numpy.ones(600), numpy.linspace(0,20,231)[:120]),
            numpy.append(2*numpy.ones(590), numpy.linspace(0,20,231)[:130]),
            numpy.append(2*numpy.ones(580),  numpy.linspace(0,20,231)[:140]),
            numpy.append(2*numpy.ones(570),  numpy.linspace(0,20,231)[:150]),
            numpy.append(2*numpy.ones(560),  numpy.linspace(0,20,231)[:160]),
            numpy.append(2*numpy.ones(550),  numpy.linspace(0,20,231)[:170]),
            numpy.append(2*numpy.ones(540),  numpy.linspace(0,20,231)[:180]),
            numpy.append(2*numpy.ones(530),  numpy.linspace(0,20,231)[:190]),
            numpy.append(2*numpy.ones(520),  numpy.linspace(0,20,231)[:200]),
            numpy.append(2*numpy.ones(510),  numpy.linspace(0,20,231)[:210]),
            numpy.append(2*numpy.ones(500),  numpy.linspace(0,20,231)[:220]),
            numpy.append(2*numpy.ones(490),   numpy.linspace(0,20,231)[:230]),
            ]
        self.timecourse_ch1 = numpy.linspace(50,30,480)
        self.timecourse_pre_ch1 = 5
        self.timecourse_sampling_steps_ch1 = numpy.arange(0,461,20).astype(int)
        self.intensity_well_ch1_2 = [
            numpy.append(5*numpy.ones(720), []),
            numpy.append(5*numpy.ones(700), numpy.linspace(50,30,480)[:20]),
            numpy.append(5*numpy.ones(680), numpy.linspace(50,30,480)[:40]),
            numpy.append(5*numpy.ones(660), numpy.linspace(50,30,480)[:60]),
            numpy.append(5*numpy.ones(640), numpy.linspace(50,30,480)[:80]),
            numpy.append(5*numpy.ones(620), numpy.linspace(50,30,480)[:100]),
            numpy.append(5*numpy.ones(600), numpy.linspace(50,30,480)[:120]),
            numpy.append(5*numpy.ones(580), numpy.linspace(50,30,480)[:140]),
            numpy.append(5*numpy.ones(560), numpy.linspace(50,30,480)[:160]),
            numpy.append(5*numpy.ones(540), numpy.linspace(50,30,480)[:180]),
            numpy.append(5*numpy.ones(520), numpy.linspace(50,30,480)[:200]),
            numpy.append(5*numpy.ones(500), numpy.linspace(50,30,480)[:220]),
            numpy.append(5*numpy.ones(480), numpy.linspace(50,30,480)[:240]),
            numpy.append(5*numpy.ones(460), numpy.linspace(50,30,480)[:260]),
            numpy.append(5*numpy.ones(440), numpy.linspace(50,30,480)[:280]),
            numpy.append(5*numpy.ones(420), numpy.linspace(50,30,480)[:300]),
            numpy.append(5*numpy.ones(400), numpy.linspace(50,30,480)[:320]),
            numpy.append(5*numpy.ones(380), numpy.linspace(50,30,480)[:340]),
            numpy.append(5*numpy.ones(360), numpy.linspace(50,30,480)[:360]),
            numpy.append(5*numpy.ones(340), numpy.linspace(50,30,480)[:380]),
            numpy.append(5*numpy.ones(320), numpy.linspace(50,30,480)[:400]),
            numpy.append(5*numpy.ones(300), numpy.linspace(50,30,480)[:420]),
            numpy.append(5*numpy.ones(280), numpy.linspace(50,30,480)[:440]),
            numpy.append(5*numpy.ones(260), numpy.linspace(50,30,480)[:460]),
            ]

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_lpa_led_set(self):
        lpa = lpadesign.LPA(name='Jennie', led_set_names=['EO_12', 'EO_20'])
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertIsInstance(lpa.led_sets[0], lpadesign.LEDSet)
        self.assertIsInstance(lpa.led_sets[1], lpadesign.LEDSet)

    def test_create_lpa_layout(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertIsInstance(lpa.led_sets[0], lpadesign.LEDSet)
        self.assertIsInstance(lpa.led_sets[1], lpadesign.LEDSet)
        self.assertEqual(lpa.led_sets[0].name, 'EO_12')
        self.assertEqual(lpa.led_sets[1].name, 'EO_20')

    def test_set_all_dc_all(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        # Change all and check
        lpa.set_all_dc(0)
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.dc, 0)
        # Change all and check
        lpa.set_all_dc(3)
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.dc, 3)

    def test_set_all_dc_ch1(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        # Change all and check
        lpa.set_all_dc(0)
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.dc, 0)
        # Change one channel and check
        lpa.set_all_dc(3, channel=0)
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.dc[:,:,0], 3)
        numpy.testing.assert_array_equal(lpa.dc[:,:,1], 0)

    def test_set_all_dc_ch2(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        # Change all and check
        lpa.set_all_dc(0)
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.dc, 0)
        # Change one channel and check
        lpa.set_all_dc(3, channel=1)
        self.assertEqual(lpa.dc.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.dc[:,:,0], 0)
        numpy.testing.assert_array_equal(lpa.dc[:,:,1], 3)

    def test_set_all_gcal_all(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        # Change all and check
        lpa.set_all_gcal(0)
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.gcal, 0)
        # Change all and check
        lpa.set_all_gcal(100)
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.gcal, 100)

    def test_set_all_gcal_ch1(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        # Change all and check
        lpa.set_all_gcal(0)
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.gcal, 0)
        # Change one channel and check
        lpa.set_all_gcal(100, channel=0)
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.gcal[:,:,0], 100)
        numpy.testing.assert_array_equal(lpa.gcal[:,:,1], 0)

    def test_set_all_gcal_ch2(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        # Change all and check
        lpa.set_all_gcal(0)
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.gcal, 0)
        # Change one channel and check
        lpa.set_all_gcal(100, channel=1)
        self.assertEqual(lpa.gcal.shape, (4, 6, 2))
        numpy.testing.assert_array_equal(lpa.gcal[:,:,0], 0)
        numpy.testing.assert_array_equal(lpa.gcal[:,:,1], 100)

    def test_set_n_steps_1(self):
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Check initial size
        self.assertEqual(lpa.intensity.shape, (1, 4, 6, 2))
        # Set all, increase size, and check
        lpa.intensity.fill(5.)
        lpa.set_n_steps(100)
        self.assertEqual(lpa.intensity.shape, (100, 4, 6, 2))
        numpy.testing.assert_array_equal(lpa.intensity, 5.)
        # Set last timepoint, extend, and check
        lpa.intensity[-1, :, :, 0] = 10.
        lpa.intensity[-1, :, :, 1] = 12.
        lpa.set_n_steps(150)
        self.assertEqual(lpa.intensity.shape, (150, 4, 6, 2))
        numpy.testing.assert_array_equal(lpa.intensity[:99,:,:,:], 5.)
        numpy.testing.assert_array_equal(lpa.intensity[100:,:,:,0], 10.)
        numpy.testing.assert_array_equal(lpa.intensity[100:,:,:,1], 12.)
        # Cut and check
        lpa.set_n_steps(120)
        self.assertEqual(lpa.intensity.shape, (120, 4, 6, 2))
        numpy.testing.assert_array_equal(lpa.intensity[:99,:,:,:], 5.)
        numpy.testing.assert_array_equal(lpa.intensity[100:,:,:,0], 10.)
        numpy.testing.assert_array_equal(lpa.intensity[100:,:,:,1], 12.)

    def test_load_dc(self):
        # Create object and attempt to load
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_dc(self.dc_file_to_load)
        # Test
        numpy.testing.assert_array_equal(lpa.dc, self.dc_to_load_exp)

    def test_load_gcal(self):
        # Create object and attempt to load
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_gcal(self.gcal_file_to_load)
        # Test
        numpy.testing.assert_array_equal(lpa.gcal, self.gcal_to_load_exp)

    def test_load_lpf(self):
        # Create object
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.dc = self.dc_to_load_exp
        lpa.gcal = self.gcal_to_load_exp
        # Initialize a different step size
        lpa.step_size = 60000
        # Load .lpf file
        lpa.load_lpf(self.lpf_file_to_load)
        # Test on step duration
        self.assertEqual(lpa.step_size, self.step_size_to_load_exp)
        # Tests on intesity array
        self.assertEqual(lpa.intensity.shape, self.intensity_to_load_exp.shape)
        numpy.testing.assert_almost_equal(lpa.intensity,
                                          self.intensity_to_load_exp)

    def test_load_files(self):
        # Create object and attempt to load
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(self.folder_to_load)
        # Test dot correction and gcal_expected+
        numpy.testing.assert_array_equal(lpa.dc, self.dc_to_load_exp)
        numpy.testing.assert_array_equal(lpa.gcal, self.gcal_to_load_exp)
        # Test on step duration
        self.assertEqual(lpa.step_size, self.step_size_to_load_exp)
        # Tests on intesity array
        self.assertEqual(lpa.intensity.shape, self.intensity_to_load_exp.shape)
        numpy.testing.assert_almost_equal(lpa.intensity,
                                          self.intensity_to_load_exp)

    def test_save_dc(self):
        # Create object and attempt to save
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.dc = self.dc_to_save
        lpa.save_dc(os.path.join(self.temp_dir, 'dc.txt'))
        # Load file and compare contents
        with open(os.path.join(self.temp_dir, 'dc.txt'), 'r') as myfile:
            file_contents=myfile.read()
        self.assertEqual(file_contents, self.dc_to_save_exp)

    def test_save_gcal(self):
        # Create object and attempt to save
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.gcal = self.gcal_to_save
        lpa.save_gcal(os.path.join(self.temp_dir, 'gcal.txt'))
        # Load file and compare contents
        with open(os.path.join(self.temp_dir, 'gcal.txt'), 'r') as myfile:
            file_contents=myfile.read()
        self.assertEqual(file_contents, self.gcal_to_save_exp)

    def test_save_lpf(self):
        # Create object
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.dc = self.dc_to_save
        lpa.gcal = self.gcal_to_save
        lpa.intensity = self.intensity_to_save
        lpa.step_size = self.step_size_to_save
        # Save
        lpa.save_lpf(os.path.join(self.temp_dir, 'program.lpf'))
        # Load file and compare with expected contents
        lpf_file_name = os.path.join(self.temp_dir, 'program.lpf')
        lpf = lpadesign.LPF(lpf_file_name)
        self.assertEqual(lpf.file_version, self.file_version_to_save_exp)
        self.assertEqual(lpf.n_channels, self.n_channels_to_save_exp)
        self.assertEqual(lpf.step_size, self.step_size_to_save_exp)
        self.assertEqual(lpf.n_steps, self.n_steps_to_save_exp)
        numpy.testing.assert_array_equal(lpf.grayscale, self.gs_to_save_exp)

    def test_save_files(self):
        # Create object
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.dc = self.dc_to_save
        lpa.gcal = self.gcal_to_save
        lpa.intensity = self.intensity_to_save
        lpa.step_size = self.step_size_to_save
        # Save
        lpa.save_files(self.temp_dir)
        # Load files and compare with expected contents
        dc_file_name = os.path.join(self.temp_dir, 'Jennie', 'dc.txt')
        with open(dc_file_name, 'r') as myfile:
            file_contents=myfile.read()
        self.assertEqual(file_contents, self.dc_to_save_exp)

        gcal_file_name = os.path.join(self.temp_dir, 'Jennie', 'gcal.txt')
        with open(gcal_file_name, 'r') as myfile:
            file_contents=myfile.read()
        self.assertEqual(file_contents, self.gcal_to_save_exp)

        lpf_file_name = os.path.join(self.temp_dir, 'Jennie', 'program.lpf')
        lpf = lpadesign.LPF(lpf_file_name)
        self.assertEqual(lpf.file_version, self.file_version_to_save_exp)
        self.assertEqual(lpf.n_channels, self.n_channels_to_save_exp)
        self.assertEqual(lpf.step_size, self.step_size_to_save_exp)
        self.assertEqual(lpf.n_steps, self.n_steps_to_save_exp)
        numpy.testing.assert_array_equal(lpf.grayscale, self.gs_to_save_exp)

    def test_set_timecourse_staggered_1(self):
        # Create object
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Set timecourse
        lpa.set_timecourse_staggered(
            intensity=self.timecourse_ch0,
            intensity_pre=self.timecourse_pre_ch0,
            sampling_steps=self.timecourse_sampling_steps_ch0,
            channel=0)
        # Test the intensity array
        self.assertEqual(lpa.intensity.shape,
                         (len(self.timecourse_ch0), 4, 6, 2))
        numpy.testing.assert_array_equal(lpa.intensity[:,:,:,1], 0)
        n_steps = lpa.intensity.shape[0]
        for row in range(4):
            for col in range(6):
                i = row*6 + col
                intensity_well = lpa.intensity[:,row,col,0]
                intensity_exp = self.intensity_well_ch0_1[i]
                numpy.testing.assert_array_equal(intensity_well, intensity_exp)

    def test_set_timecourse_staggered_2(self):
        # Create object
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Set long experiment duration
        lpa.set_n_steps(720)
        # Set timecourse
        lpa.set_timecourse_staggered(
            intensity=self.timecourse_ch0,
            intensity_pre=self.timecourse_pre_ch0,
            sampling_steps=self.timecourse_sampling_steps_ch0,
            channel=0)
        # Test the intensity array
        self.assertEqual(lpa.intensity.shape, (720, 4, 6, 2))
        numpy.testing.assert_array_equal(lpa.intensity[:,:,:,1], 0)
        n_steps = lpa.intensity.shape[0]
        for row in range(4):
            for col in range(6):
                i = row*6 + col
                intensity_well = lpa.intensity[:,row,col,0]
                intensity_exp = self.intensity_well_ch0_2[i]
                numpy.testing.assert_array_equal(intensity_well, intensity_exp)

    def test_set_timecourse_staggered_3(self):
        # Create object
        lpa = lpadesign.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        # Set long experiment duration
        lpa.set_n_steps(720)
        # Set timecourse
        lpa.set_timecourse_staggered(
            intensity=self.timecourse_ch0,
            intensity_pre=self.timecourse_pre_ch0,
            sampling_steps=self.timecourse_sampling_steps_ch0,
            channel=0)
        lpa.set_timecourse_staggered(
            intensity=self.timecourse_ch1,
            intensity_pre=self.timecourse_pre_ch1,
            sampling_steps=self.timecourse_sampling_steps_ch1,
            channel=1)
        # Test the intensity array
        self.assertEqual(lpa.intensity.shape, (720, 4, 6, 2))
        n_steps = lpa.intensity.shape[0]
        for row in range(4):
            for col in range(6):
                i = row*6 + col
                # Channel 0
                intensity_well = lpa.intensity[:,row,col,0]
                intensity_exp = self.intensity_well_ch0_2[i]
                numpy.testing.assert_array_equal(intensity_well, intensity_exp)
                # Channel 1
                intensity_well = lpa.intensity[:,row,col,1]
                intensity_exp = self.intensity_well_ch1_2[i]
                numpy.testing.assert_array_equal(intensity_well, intensity_exp)

