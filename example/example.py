import numpy
import lpaprogram

lpaprogram.LED_DATA_PATH = "../test/test_lpa_files/led-archives"

if __name__ == "__main__":
    # Load LPAs
    lpa = lpaprogram.LPA(name='Jennie',  led_set_names=['EO_12', 'EO_20'])
    # Step size is one minute
    lpa.step_size = 60000
    # Exp time is 8 hours
    lpa.set_n_steps(60*8)

    # Set gcal values
    lpa.set_all_gcal(255, channel=0)
    lpa.set_all_gcal(255, channel=1)

    # Define Light intensity signal
    # Green light: ramp to 100
    # Red light: constant at 20
    gls = numpy.logspace(-1, 2, 360)
    gls_pre = 0
    rls = 20
    sampling_steps = numpy.arange(24)*15
    # Set intensity data
    lpa.set_timecourse_staggered(intensity=gls,
                                 intensity_pre=gls_pre,
                                 sampling_steps=sampling_steps,
                                 channel=0)
    lpa.intensity[:,:,:,1]=rls
    # Adjust dc values
    lpa.optimize_dc(channel=0, uniform=True)
    lpa.optimize_dc(channel=1, uniform=True)
    lpa.discretize_intensity()

    # Save files
    lpa.save_files()

    # Plot light signal per well
    lpa.plot_intensity(channel=0,
                       file_name='channel_0.png',
                       xunits='min')
    lpa.plot_intensity(channel=1,
                       file_name='channel_1.png',
                       xunits='min')
