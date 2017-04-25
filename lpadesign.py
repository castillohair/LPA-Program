"""
Tools for designing LPA experiments.

"""

# Versions should comply with PEP440. For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '0.1.0'

import numpy
import os
import pandas
import struct
import random

LED_DATA_PATH = ""

class LPFFile():
    """
    Class that represents an lpf file.

    Properties:
    - file_version      -- The lpf file version
    - total_nchannels   -- The total number of channels (i.e. LEDs) in the
                            device.
    - step_size         -- The size of the time step, in milliseconds.
    - nsteps            -- The number of time steps in the lpf file.
    - well_nchannels    -- The number of channels (LEDs) in each well.
    - grayscale         -- A numpy array, with dimensions (nsteps,
                            total_nchannels/well_nchannels, well_nchannels)
                            with the grayscale intensities of the LPF

    """

    def __init__(self,
                 file_name=None,
                 well_nchannels=2):

        # Save channels per well
        self.well_nchannels = well_nchannels

        # Initialize properties
        self.file_version = 1
        self.total_nchannels = None
        self.step_size = None
        self.nsteps = None
        self.grayscale = None

        # Open file name
        if file_name is not None:
            self.load(file_name)

    def load(self, file_name):
        # Open file
        f = open(file_name, 'rb')

        # Information reading from this file will be made inside a try block,
        # to free resources in case anything goes wrong.
        try:
            # Header is 32 bytes
            # First 4 bytes are the file version
            self.file_version = struct.unpack('<I', f.read(4))[0]

            # What to do if file version is 1.0
            if self.file_version == 1:
                # Next 4 bytes are the total number of channels
                self.total_nchannels = struct.unpack('<I', f.read(4))[0]
                # Next 4 bytes are the step size in ms
                self.step_size = struct.unpack('<I', f.read(4))[0]
                # Next 4 bytes are the number of steps
                self.nsteps = struct.unpack('<I', f.read(4))[0]

                # Read grayscale
                # Calculate size of intensity block
                number_words_data = self.total_nchannels*self.nsteps
                # Read data block
                data = numpy.memmap(
                    f,
                    dtype=numpy.dtype('<u2'),
                    mode='r',
                    offset=32,
                    shape=(number_words_data,),
                    order='C')
                data = numpy.array(data)
                # Resize to get grayscale values
                self.grayscale = data.reshape((
                    self.nsteps,
                    self.total_nchannels/self.well_nchannels,
                    self.well_nchannels))

            else:
                raise NotImplementedError("LPF file version {} not recognized"
                    .format(self.file_version))

        finally:
            f.close()

    def save(self, file_name):
        # Open file for writing
        f = open(file_name, 'wb')

        # Use a try block to free resources in case anything goes wrong
        try:
            # Header is 32 bytes
            # First 4 bytes are the file version
            f.write(struct.pack('<I', self.file_version))

            # What to do if file version is 1.0
            if self.file_version == 1:
                # Next 4 bytes are the total number of channels
                f.write(struct.pack('<I', self.total_nchannels))
                # Next 4 bytes are the step size in ms
                f.write(struct.pack('<I', self.step_size))
                # Next 4 bytes are the number of steps
                f.write(struct.pack('<I', self.nsteps))
                # Write 16 more empty bytes
                f.write(struct.pack('<IIII', 0, 0, 0, 0))
                # Saturate grayscale at 4095 and save
                gs = self.grayscale.astype(numpy.uint16)
                gs[gs > 4095] = 4095
                gs.tofile(f)

            else:
                raise NotImplementedError("LPF file version {} not recognized"
                    .format(self.file_version))

        finally:
            f.close()

class LEDSet(object):
    def __init__(self, name, lpa_name, channel, nrows=4, ncols=6):
        # Store plate dimensions
        self.nrows = nrows
        self.ncols = ncols
        # Transform channel to numerical value
        if channel in [0, 'c1', 'Top']:
            channel = 0
        elif channel in [1, 'c2', 'Bot', 'Bottom']:
            channel = 1
        else:
            raise ValueError('channel not valid')
        # Store arguments
        self.name = name
        self.lpa_name = lpa_name
        self.channel = channel
        # Construct name of file with led set data
        file_name = os.path.join(
            LED_DATA_PATH,
            name,
            "{}_c{}".format(lpa_name, channel + 1),
            "{}_{}_c{}.xlsx".format(name, lpa_name, channel + 1))
        # Open calibration data
        self.calibration_data = pandas.read_excel(file_name, 'Sheet1')
        # Check dimensions
        if len(self.calibration_data) != (self.nrows*self.ncols):
            raise ValueError("calibration data does not have the expected " + \
                "dimensions")

    def get_intensity(self, row, col, gs, dc, gcal=255):
        # If row is None, use all wells
        if row is None:
            well = numpy.arange(self.nrows*self.ncols) + 1
        else:
            # Transform (row, col) pair into well number
            row = numpy.array(row)
            col = numpy.array(col)
            well = (row - 1)*self.ncols + col
        # Get info for relevant wells
        led_data = self.calibration_data[self.calibration_data['Well']\
            .isin(well)]
        # Get intensity at measured conditions
        measured_dc = led_data['DC'].values.astype(float)
        measured_gcal = led_data['GS Cal'].values.astype(float)
        measured_intensity = led_data['Intensity (umol/m2/s)']\
            .values.astype(float)
        # Calculate intensity
        intensity = measured_intensity * (dc/measured_dc) * \
                                         (gcal/measured_gcal) * \
                                         (gs/4095.)

        return intensity

    def get_grayscale(self, row, col, intensity, dc, gcal=255):
        # If row is None, use all wells
        if row is None:
            well = numpy.arange(self.nrows*self.ncols) + 1
        else:
            # Transform (row, col) pair into well number
            row = numpy.array(row)
            col = numpy.array(col)
            well = (row - 1)*self.ncols + col
        # Get info for relevant wells
        led_data = self.calibration_data[self.calibration_data['Well']\
            .isin(well)]
        # Get intensity at measured conditions
        measured_dc = led_data['DC'].values.astype(float)
        measured_gcal = led_data['GS Cal'].values.astype(float)
        measured_intensity = led_data['Intensity (umol/m2/s)']\
            .values.astype(float)
        # Calculate grayscale value
        gs = 4095. * (intensity/measured_intensity) * \
                     (measured_dc/dc) * \
                     (measured_gcal/gcal)
        gs = gs.astype(numpy.uint16)
        if numpy.any(gs > 4095):
            raise ValueError("not possible to generate requested intensity " + \
                "with provided dc value. ")

        return gs

class LPA(object):
    def __init__(self,
                 name,
                 led_set_names,
                 nrows=4,
                 ncols=6,
                 nchannels=2,
                 step_size=1000):
        # Store name
        self.name = name
        # Store dimensions
        self.nrows = nrows
        self.ncols = ncols
        self.nchannels = nchannels
        self.step_size = step_size
        # Check length of led_set_names
        if len(led_set_names) != nchannels:
            raise ValueError("led_set_names should have {} elements"\
                .format(nchannels))
        # Initialize led sets
        self.led_sets = []
        for i, led_set_name in enumerate(led_set_names):
            if led_set_name is None:
                self.led_sets.append(None)
            else:
                self.led_sets.append(LEDSet(name=led_set_name,
                                            lpa_name=name,
                                            channel=i,
                                            nrows=nrows,
                                            ncols=ncols))
        # Initialize dc and gcal arrays
        self.dc = numpy.zeros((self.nrows,
                               self.ncols,
                               self.nchannels), dtype=int)
        self.gcal = numpy.zeros((self.nrows,
                                 self.ncols,
                                 self.nchannels), dtype=int)
        # Intensity is a 4D array with dimensions [step, row, col, channel]
        self.intensity = numpy.zeros((1,
                                      self.nrows,
                                      self.ncols,
                                      self.nchannels))

    def set_all_dc(self, value, channel=None):
        if channel is None:
            self.dc = numpy.ones((self.nrows,
                                  self.ncols,
                                  self.nchannels), dtype=int)*value
        else:
            self.dc[:,:,channel] = value

    def set_all_gcal(self, value, channel=None):
        if channel is None:
            self.gcal = numpy.ones((self.nrows,
                                    self.ncols,
                                    self.nchannels), dtype=int)*value
        else:
            self.gcal[:,:,channel] = value

    def set_nsteps(self, nsteps):
        if nsteps > self.intensity.shape[0]:
            # To add steps, repeat the last intensity value
            steps = numpy.expand_dims(self.intensity[-1,:,:,:], axis=0)
            steps = numpy.repeat(steps,
                                 nsteps - self.intensity.shape[0],
                                 axis=0)
            self.intensity = numpy.append(self.intensity, steps, axis=0)
        elif nsteps < self.intensity.shape[0]:
            # To eliminate steps, we will just slice
            self.intensity = self.intensity[nsteps,:,:,:]

    def set_intensity_staggered(self,
                                intensity,
                                intensity_pre,
                                sampling_steps,
                                channel,
                                rows=None,
                                cols=None):
        # Populate row and col arrays if necessary
        if rows is None:
            rows = numpy.repeat(numpy.arange(self.nrows), self.ncols)
            cols = numpy.tile(numpy.arange(self.ncols), self.nrows)
        # Check matching dimensions
        if len(rows) != len(cols):
            raise ValueError("rows and cols should have the same length")
        # Check that the number of sampling steps matches the number of rows and
        # columns
        if len(rows) != len(sampling_steps):
            raise ValueError("Number of sampling steps should match the number"\
                " of wells")
        # Expand the intensity array if necessary
        if self.intensity.shape[0] < len(intensity):
            self.set_nsteps(len(intensity))
        # Calculate start of induction step
        nsteps = self.intensity.shape[0]
        start_steps = nsteps - sampling_steps
        # Populate intensity array
        for row, col, start_step in zip(rows, cols, start_steps):
            intensity_well = numpy.ones(nsteps)*intensity_pre
            intensity_well[start_step:] = intensity[:len(intensity_well) - start_step]
            self.intensity[:, row, col, channel] = intensity_well

    def save_dc(self, file_name):
        # Flatten dc array
        dc = self.dc.reshape((self.nrows, self.nchannels*self.ncols))
        # Generate string to save
        s = ''
        for dc_row in dc:
            s += "\t".join(dc_row.astype(str))
            s += "\n"
        # Save
        f = open(file_name, "w")
        f.write(s)
        f.close()

    def save_gcal(self, file_name):
        # Flatten gcal array
        gcal = self.gcal.reshape((self.nrows, self.nchannels*self.ncols))
        # Generate string to save
        s = ''
        for gcal_row in gcal:
            s += "\t".join(gcal_row.astype(str))
            s += "\n"
        # Save
        f = open(file_name, "w")
        f.write(s)
        f.close()

    def save_lpf(self, file_name):
        nsteps = self.intensity.shape[0]
        # Initialize grayscale array
        gs = numpy.zeros((nsteps,
                          self.nrows*self.ncols,
                          self.nchannels), dtype=int)
        # Convert intensities to grayscale values
        for channel in range(self.nchannels):
            dc_channel = self.dc[:,:,channel].flatten()
            gcal_channel = self.gcal[:,:,channel].flatten()
            for step in range(nsteps):
                try:
                    gs[step, :, channel] = self.led_sets[channel].get_grayscale(
                        row=None,
                        col=None,
                        intensity=self.intensity[step, :, :, channel].flatten(),
                        dc=dc_channel,
                        gcal=gcal_channel,
                        )
                except ValueError as e:
                    print e.args
                    e.args = ("on step {}, channel {}: ".format(
                        step,
                        channel) + e.args[0],)
                    raise
        # Create LPFFile object
        lpf_file = LPFFile(well_nchannels=self.nchannels)
        lpf_file.total_nchannels = self.nchannels*self.nrows*self.ncols
        lpf_file.step_size = self.step_size
        lpf_file.nsteps = nsteps
        lpf_file.grayscale = gs
        lpf_file.save(file_name)

    def save_files(self, path='.'):
        # Add name of lpa to path
        path = os.path.join(path, self.name)
        # Create folder if necessary
        if not os.path.exists(path):
            os.makedirs(path)
        # Save dc, gcal, and lpf files
        self.save_dc(os.path.join(path, 'dc.txt'))
        self.save_gcal(os.path.join(path, 'gcal.txt'))
        self.save_lpf(os.path.join(path, 'program.lpf'))
        # Save additional empty file with LPA's name
        open(os.path.join(path, self.name + ".txt"), 'w').close()

