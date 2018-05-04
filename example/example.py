import numpy
import lpaprogram

lpaprogram.LED_CALIBRATION_PATH = "Calibration Data"

if __name__ == "__main__":
    # Load LPA
    lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
    # Step size is one minute
    lpa.step_size = 60000
    # Exp time is 8 hours
    lpa.set_n_steps(60*8)

    # Set gcal values
    lpa.set_all_gcal(255, channel=0)
    lpa.set_all_gcal(255, channel=1)

    # Define Light intensity signal
    # Channel 1: ramp to 50
    # Channel 2: constant at 20
    gls = numpy.logspace(-1, numpy.log10(50), 360)
    gls_pre = 0
    rls = 20
    sampling_steps = numpy.arange(24)*15
    # Set intensity data
    lpa.set_timecourse_staggered(intensity=gls,
                                 intensity_pre=gls_pre,
                                 sampling_steps=sampling_steps,
                                 channel=0)
    lpa.intensity[:,:,:,1]=rls

    # Discretize intensity
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
