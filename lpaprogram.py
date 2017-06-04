"""
Generate programs for a Light Plate Apparatus (LPA).

"""

# Versions should comply with PEP440. For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '1.0.0b1'

import os
import random
import struct

import numpy
import pandas
from matplotlib import pyplot

LED_DATA_PATH = ""

class LPF(object):
    """
    Class that represents a light program file (.lpf).

    Parameters
    ----------
    file_name : str, optional
        If present, the object will be initialized with data from an .lpf
        file specified by this argument.

    Attributes
    ----------
    file_version : int
        lpf file version.
    n_channels : int
        Total number of channels (i.e. LEDs) in the device.
    step_size : int
        Size of the time step, in milliseconds.
    n_steps : int
        Number of time steps.
    grayscale : array
        Grayscale LED intensities. Its dimensions are ``(n_steps,
        n_channels)``.

    Methods
    -------
    load
        Load data from an lpf file.
    save
        Save data into an lpf file.

    """

    def __init__(self,
                 file_name=None):

        # Initialize properties
        self.file_version = 1
        self.n_channels = None
        self.step_size = None
        self.n_steps = None
        self.grayscale = None

        # Open file name
        if file_name is not None:
            self.load(file_name)

    def load(self, file_name):
        """
        Load data from an lpf file.

        Parameters
        ----------
        file_name : str
            Name of the file to load.

        """
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
                self.n_channels = struct.unpack('<I', f.read(4))[0]
                # Next 4 bytes are the step size in ms
                self.step_size = struct.unpack('<I', f.read(4))[0]
                # Next 4 bytes are the number of steps
                self.n_steps = struct.unpack('<I', f.read(4))[0]

                # Read grayscale
                # Calculate size of intensity block
                number_words_data = self.n_channels*self.n_steps
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
                    self.n_steps,
                    self.n_channels))

            else:
                raise NotImplementedError("LPF file version {} not recognized"
                    .format(self.file_version))

        finally:
            f.close()

    def save(self, file_name):
        """
        Save data into an lpf file.

        Parameters
        ----------
        file_name : str
            Name of the file to save.

        """
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
                f.write(struct.pack('<I', self.n_channels))
                # Next 4 bytes are the step size in ms
                f.write(struct.pack('<I', self.step_size))
                # Next 4 bytes are the number of steps
                f.write(struct.pack('<I', self.n_steps))
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
    """
    Object that represents an LED set.

    Spectral measurements of this LED set in a specific LPA have been
    conducted with specific values of dot correction (dc) and grayscale
    calibration (gcal). These measurements should be saved into an Excel
    table that is loaded during the object's creation. These measurements
    are used to convert from light intensity values in umol/m^2/s into
    grayscale values at the specified dc and gcal values, and viceversa.

    Parameters
    ----------
    name : str
        Name of LED set.
    file_name : str
        Name of the Excel file in which spectral measurements are stored.

    Attributes
    ----------
    name : str
        Name of LED set.
    lpa_name : str
        Name of the LPA in which spectral measurements were conducted.
    n_rows : int
        Number of rows in the LPA.
    n_cols : int
        Number of cols in the LPA.
    channel : int
        Channel of the LPA in which the LED set is located.
    spectral_data : DataFrame
        A table with the LED set spectral data.

    Methods
    -------
    get_intensity
        Calculate intensity in umol/m^2/s from grayscale values.
    get_grayscale
        Calculate grayscale values from intensity values in umol/m^2/s.

    """
    def __init__(self, name, file_name):
        # Store name
        self.name = name
        # Load spectral data
        self.spectral_data = pandas.read_excel(file_name,
                                               'Sheet1',
                                               index_col='Well')
        # Extract LPA information
        self.lpa_name = self.spectral_data['LPA'].iloc[0]
        self.n_rows = self.spectral_data['Row'].max()
        self.n_cols = self.spectral_data['Col'].max()
        channel = self.spectral_data['Channel'].iloc[0]
        if channel in [1, 'c1', 'Top']:
            self.channel = 0
        elif channel in [2, 'c2', 'Bot', 'Bottom']:
            self.channel = 1
        else:
            raise ValueError("channel not recognized")
        # Sanity checks
        if not (self.spectral_data['LPA']==self.lpa_name).all():
            raise ValueError("LPA name is not consistent in spectral data")
        if not (self.spectral_data['Channel']==channel).all():
            raise ValueError("channel is not consistent in spectral data")
        if len(self.spectral_data) != (self.n_rows*self.n_cols):
            raise ValueError("spectral data does not have the expected " + \
                "dimensions")

    def get_intensity(self, gs, dc, gcal=255, row=None, col=None):
        """
        Calculate intensity in umol/m^2/s from grayscale values.

        Parameters can be arrays or single numbers, and these can be mixed.
        All arrays should have the same dimensions. If either ``row`` or
        ``column`` are None, all wells are used.

        Parameters
        ----------
        gs : array
            Grayscale values to convert.
        dc : array
            Dot-correction values.
        gcal : array, optional
            Grayscale calibration values.
        row : array, optional
            Row positions of each grayscale value to convert, zero-indexed.
        col : array, optional
            Column positions of each grayscale value to convert,
            zero-indexed.

        Returns
        -------
        array
            The intensities of each well in umol/m^2/s.

        """
        # If row is None, use all wells
        if (row is None) or (col is None):
            well = numpy.arange(self.n_rows*self.n_cols)
        else:
            # Transform (row, col) pair into well number
            row = numpy.atleast_1d(row)
            col = numpy.atleast_1d(col)
            well = row*self.n_cols + col
        # Get info for relevant wells
        led_data = self.spectral_data.loc[well + 1]
        # Get intensity at measured conditions
        measured_dc = led_data['DC'].values.astype(float)
        measured_gcal = led_data['GS Cal'].values.astype(float)
        measured_intensity = led_data['Intensity (umol/m2/s)']\
            .values.astype(float)
        # Calculate intensity
        dc = numpy.array(dc)
        gcal = numpy.array(gcal)
        gs = numpy.array(gs)
        intensity = measured_intensity * (dc/measured_dc) * \
                                         (gcal/measured_gcal) * \
                                         (gs/4095.)

        return intensity

    def get_grayscale(self, intensity, dc, gcal=255, row=None, col=None):
        """
        Calculate grayscale values to achieve the specified intensities.

        Parameters can be arrays or single numbers, and these can be mixed.
        All arrays should have the same dimensions. If either ``row`` or
        ``column`` are None, all wells are used.

        The returned grayscale values are rounded to the nearest integer,
        so the resulting intensities may not exactly be the ones specified.
        If the resulting grayscale value is higher than 4095 for any well,
        this function raises an error.

        Parameters
        ----------
        intensity : array
            The intensities of each well in umol/m^2/s.
        dc : array
            Dot-correction values.
        gcal : array, optional
            Grayscale calibration values.
        row : array, optional
            Row positions of each intensity value to convert, zero-indexed.
        col : array, optional
            Column positions of each intensity value to convert,
            zero-indexed.

        Returns
        -------
        array
            Grayscale values to achieve the specified intensities.

        """
        # If row is None, use all wells
        if (row is None) or (col is None):
            well = numpy.arange(self.n_rows*self.n_cols)
        else:
            # Transform (row, col) pair into well number
            row = numpy.atleast_1d(row)
            col = numpy.atleast_1d(col)
            well = row*self.n_cols + col
        # Get info for relevant wells
        led_data = self.spectral_data.loc[well + 1]
        # Get intensity at measured conditions
        measured_dc = led_data['DC'].values.astype(float)
        measured_gcal = led_data['GS Cal'].values.astype(float)
        measured_intensity = led_data['Intensity (umol/m2/s)']\
            .values.astype(float)
        # Calculate grayscale value
        dc = numpy.array(dc)
        gcal = numpy.array(gcal)
        intensity = numpy.array(intensity)
        gs = 4095. * (intensity/measured_intensity) * \
                     (measured_dc/dc) * \
                     (measured_gcal/gcal)
        gs = numpy.round(gs).astype(numpy.uint16)
        if numpy.any(gs > 4095):
            raise ValueError("not possible to generate requested intensity " + \
                "with provided dc value. ")

        return gs

    def discretize_intensity(self, intensity, dc, gcal=255, row=None, col=None):
        """
        Discretize intensity values.

        At constant dc and gcal values, intensities are represented by
        integer grayscale values ranging from 0 to 4095. Because of this
        finite resolution, only a discrete number of intensity values are
        possible for any given well. This function takes this into account
        to convert the provided intensity values to the closest values that
        are possible at the given dc and gcal values.

        Parameters
        ----------
        intensity : array
            The intensities of each well in umol/m^2/s.
        dc : array
            Dot-correction values.
        gcal : array, optional
            Grayscale calibration values.
        row : array, optional
            Row positions of each intensity value to discretize,
            zero-indexed.
        col : array, optional
            Column positions of each intensity value to discretize,
            zero-indexed.

        Returns
        -------
        array
            Discretized intensity values

        Notes
        -----
        This function achieves discretization by first converting the
        provided intensity values to integer grayscale values using
        `get_grayscale`, and then converting those back to intensity values
        with `get_intensity`.

        """
        # Convert to grayscale, then to intensity
        gs = self.get_grayscale(intensity, dc, gcal, row, col)
        return self.get_intensity(gs, dc, gcal, row, col)

    def optimize_dc(self,
                    intensity,
                    gcal=255,
                    min_dc=1,
                    uniform=False,
                    row=None,
                    col=None,
                    ):
        """
        Get the lowest dc value so that a specified intensity is possible.

        At constant dc and gcal values, intensities are represented by
        integer grayscale values ranging from 0 to 4095. Therefore, at a
        given dc value there is a highest possible intensity. While larger
        dc values allow for larger intensities, resolution is lost, as each
        grayscale step represents a larger increase in intensity as well.
        This function calculates the lowest dc required to achieve a
        certain intensity, which also maximizes the resolution for lower
        intensities.

        Parameters
        ----------
        intensity : array
            The intensities of each well in umol/m^2/s.
        gcal : array, optional
            Grayscale calibration values.
        min_dc : array, optional
            Minimum dc value to return.
        uniform : bool, optional
            Whether all returned dc values should be the same.
        row : array, optional
            Row positions of each intensity value, zero-indexed.
        col : array, optional
            Column positions of each intensity value, zero-indexed.

        Returns
        -------
        array
            Optimized dot correction values.

        """
        # If row is None, use all wells
        if (row is None) or (col is None):
            well = numpy.arange(self.n_rows*self.n_cols)
        else:
            # Transform (row, col) pair into well number
            row = numpy.atleast_1d(row)
            col = numpy.atleast_1d(col)
            well = row*self.n_cols + col
        # Get info for relevant wells
        led_data = self.spectral_data.loc[well + 1]
        # Get intensity at measured conditions
        measured_dc = led_data['DC'].values.astype(float)
        measured_gcal = led_data['GS Cal'].values.astype(float)
        measured_intensity = led_data['Intensity (umol/m2/s)']\
            .values.astype(float)
        # Calculate dc value
        # We calculate the dot correction value required to achieve the desired
        # intensity with 4095 grayscale, and then round up.
        gcal = numpy.array(gcal)
        intensity = numpy.array(intensity)
        dc = measured_dc * (intensity/measured_intensity) * \
                           (measured_gcal/gcal)
        dc = numpy.ceil(dc).astype(numpy.uint16)
        # Saturate
        dc = numpy.maximum(dc, min_dc)
        # Make uniform if necessary
        if uniform:
            dc.fill(numpy.max(dc))
        # Verify that the dc values are feasible
        if numpy.any(dc > 63):
            raise ValueError("not possible to generate requested intensity")

        return dc

class LPA(object):
    """
    Object that represents an LPA with associated LED sets.

    This object manages the intensity, dot correction, and grayscale
    calibration values of an LPA, and can convert these from/into the text/
    binary files that are directly used by an LPA.

    This object works with intensity values in umol/m^2/s, and uses LEDSet
    objects to convert these into grayscale values before saving LPA files.
    The names of the LED sets should be specified during the object's
    creation. Spectral measurements of LED sets are assumed to be present
    in the folder "{LED_DATA_PATH} / {led_set_name} / {lpa_name}_{channel}"
    in a file named "{led_set_name}_{lpa_name}_{channel}.xlsx".

    Alternatively, LED layouts can be specified instead of LED set names.
    A layout is a group of LED sets that use similar LEDs, but with each
    LED set calibrated against different LPAs. Layouts can have more
    generic and descriptive names (e.g. "660nm LEDs"), whereas each LED set
    calibrated against an LPA needs to have a unique name. To use layouts,
     a file "led_archives.xlsx" should be present in ``LED_DATA_PATH``.
    This file specifies the mapping from layout name and LPA name to the
    appropriate LED set name.

    Properties
    ----------
    name : str
        Name of LPA.
    led_set_names : list, optional
        LED set names for each channel. Either this or `layout_names`
        should be specified.
    layout_names : list, optional
        Layout names for each channel. Either this or `led_set_names`
        should be specified.

    Attributes
    ----------
    name : str
        Name of the LPA.
    led_sets : list
        LEDSet objects for each channel.
    n_rows : int
        Number of rows in the LPA.
    n_cols : int
        Number of cols in the LPA.
    n_channels : int
        Number of channels (LEDs per well) in the LPA.
    step_size : int
        Duration of each time step in `intensity` array, in milliseconds.
    dc : array
        Array of size (n_rows, n_cols, n_channels) with dot correction
        values.
    gcal : array
        Array of size (n_rows, n_cols, n_channels) with grayscale
        calibration values.
    intensity : array
        Array of size (n_steps, n_rows, n_cols, n_channels) with light
        intensity values for each LED, in umol/m^2/s.

    Methods
    -------
    set_all_dc
        Set all dc values for a specific channel or for all of them.
    set_all_gcal
        Set all gcal values for a specific channel or for all of them.
    set_n_steps
        Resize the intensity array to a specific number of time steps.
    load_dc
        Load dc values from a tab-separated text file.
    load_gcal
        Load gcal values from a tab-separated text file.
    load_lpf
        Load intensity values from a binary .lpf file.
    load_files
        Load a set of dc, gcal, and lpf files in a specified folder.
    save_dc
        Save dc values in a tab-separated text file.
    save_gcal
        Save gcal values in a tab-separated text file.
    save_lpf
        Save grayscale values in a binary .lpf file.
    save_files
        Save dc, gcal, and .lpf files from the contents of this object.
    set_timecourse_staggered
        Set an intensity timecourse on many wells, with staggered delays.
    discretize_intensity
        Discretize the values in the intensity array.
    optimize_dc
        Get the lowest dc value so that a specified intensity is possible.
    plot_intensity
        Plot the light intensity for each well in the LPA.

    """
    def __init__(self,
                 name,
                 led_set_names=None,
                 layout_names=None):

        # Store name
        self.name = name

        # Check that both led_set_names and layout_names are not None
        if (layout_names is None) and (led_set_names is None):
            raise ValueError('layout_names or led_set_names should be '
                'specified')

        # Obtain LED set names from layout names
        if layout_names is not None:
            # More than two channels are not supported here
            if len(layout_names) > 2:
                raise NotImplementedError("more than 2 channels not supported")
            # Load layout table
            layout_table = pandas.read_excel(os.path.join(LED_DATA_PATH,
                                                          'led_archives.xlsx'))
            # Obtain led set names
            led_set_names = []
            for i, layout in enumerate(layout_names):
                # Get data for corresponding row in layout table
                channel = ['Top', 'Bot'][i]
                layout_row = layout_table[(layout_table['LPA']==name) &\
                                          (layout_table['Channel']==channel) &\
                                          (layout_table['Layout']==layout)]
                # Check for more or less than one hit
                if len(layout_row) > 1:
                    raise ValueError("more than one rows with LPA name {},"
                        " Channel {}, Layout {} in led_archives.xlsx".format(
                            name, channel, layout))
                elif len(layout_row) < 1:
                    raise ValueError("no layout data for LPA name {},"
                        " Channel {}, Layout {} in led_archives.xlsx".format(
                            name, channel, layout))
                # Accumulate
                led_set_names.append(layout_row.iloc[0]['Archive ID'])

        # Initialize led sets
        self.led_sets = []
        for i, led_set_name in enumerate(led_set_names):
            file_name = os.path.join(
                LED_DATA_PATH,
                led_set_name,
                "{}_c{}".format(name, i+1),
                "{}_{}_c{}.xlsx".format(led_set_name, name, i+1))
            self.led_sets.append(LEDSet(name=led_set_name,
                                        file_name=file_name))

        # Store dimensions
        self.n_rows = self.led_sets[0].n_rows
        self.n_cols = self.led_sets[0].n_cols
        self.n_channels = len(self.led_sets)

        # Consistency checks on all led sets
        for led_set in self.led_sets:
            if self.name != led_set.lpa_name:
                print "LPA name does not match for LED set {}".format(
                    led_set.name)
            if self.n_rows != led_set.n_rows:
                print "number of rows does not match for LED set {}".format(
                    led_set.name)
            if self.n_cols != led_set.n_cols:
                print "number of columns does not match for LED set {}".format(
                    led_set.name)

        # Initialize step size in ms
        self.step_size = 1000
        # Initialize dc and gcal arrays
        self.dc = numpy.zeros((self.n_rows,
                               self.n_cols,
                               self.n_channels), dtype=int)
        self.gcal = numpy.ones((self.n_rows,
                                 self.n_cols,
                                 self.n_channels), dtype=int)*255
        # Intensity is a 4D array with dimensions [step, row, col, channel]
        self.intensity = numpy.zeros((1,
                                      self.n_rows,
                                      self.n_cols,
                                      self.n_channels))

    def set_all_dc(self, value, channel=None):
        """
        Set all dc values for a specific channel or for all of them.

        Parameters
        ----------
        value : int
            Desired dot correction value, from 0 to 63.
        channel : int, optional
            Channel for which the specified dot correction value should be
            assigned. If None, assign to all channels.

        """
        if channel is None:
            self.dc.fill(value)
        else:
            self.dc[:,:,channel] = value

    def set_all_gcal(self, value, channel=None):
        """
        Set all gcal values for a specific channel or for all of them.

        Parameters
        ----------
        value : int
            Desired grayscale calibration value, from 0 to 255.
        channel : int, optional
            Channel for which the specified dot correction value should be
            assigned. If None, assign to all channels.

        """
        if channel is None:
            self.gcal.fill(value)
        else:
            self.gcal[:,:,channel] = value

    def set_n_steps(self, n_steps):
        """
        Resize the intensity array to a specific number of time steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to resize the intensity array to. If `n_steps`
            is lower than the current length of `intensity`, the latter
            values will be discarded. If `n_steps` is larger, the last
            timepoint of `intensity` will be repeated.

        """
        if n_steps > self.intensity.shape[0]:
            # To add steps, repeat the last intensity value
            steps = numpy.expand_dims(self.intensity[-1,:,:,:], axis=0)
            steps = numpy.repeat(steps,
                                 n_steps - self.intensity.shape[0],
                                 axis=0)
            self.intensity = numpy.append(self.intensity, steps, axis=0)
        elif n_steps < self.intensity.shape[0]:
            # To eliminate steps, we will just slice
            self.intensity = self.intensity[:n_steps,:,:,:]

    def load_dc(self, file_name):
        """
        Load dc values from a tab-separated text file.

        Parameters
        ----------
        file_name : str
            Name of the file to load.

        """
        with open(file_name, 'r') as myfile:
            file_contents=myfile.read()
        self.dc = numpy.array([int(si) for si in file_contents.split()],
                              dtype=int)
        self.dc.resize(self.n_rows, self.n_cols, self.n_channels)

    def load_gcal(self, file_name):
        """
        Load gcal values from a tab-separated text file.

        Parameters
        ----------
        file_name : str
            Name of the file to load.

        """
        with open(file_name, 'r') as myfile:
            file_contents=myfile.read()
        self.gcal = numpy.array([int(si) for si in file_contents.split()],
                                dtype=int)
        self.gcal.resize(self.n_rows, self.n_cols, self.n_channels)

    def load_lpf(self, file_name):
        """
        Load intensity values from a binary .lpf file.

        Intensity values are calculated from the grayscale values found in
        the .lpf file using the associated LEDSet objects, and the dc and
        gcal arrays.

        Parameters
        ----------
        file_name : str
            Name of the file to load.

        """
        # Load light program file
        lpf = LPF(file_name)
        # Check dimensions
        if lpf.n_channels != self.n_rows*self.n_cols*self.n_channels:
            raise ValueError("unexpected number of channels in light program "
                "file")
        # Populate intensity array
        self.set_n_steps(lpf.n_steps)
        gs = numpy.resize(lpf.grayscale, (lpf.n_steps,
                                          self.n_rows,
                                          self.n_cols,
                                          self.n_channels))
        for step in range(lpf.n_steps):
            for channel in range(self.n_channels):
                # Get intensities from LED set
                intensity_sch = self.led_sets[channel].get_intensity(
                    gs=gs[step,:,:,channel].flatten(),
                    dc=self.dc[:,:,channel].flatten(),
                    gcal=self.gcal[:,:,channel].flatten())
                # Resize and add to intensity array
                intensity_sch.resize(self.n_rows, self.n_cols)
                self.intensity[step, :, :, channel] = intensity_sch
        # Set step size
        self.step_size = lpf.step_size

    def load_files(self, path):
        """
        Load a set of dc, gcal, and lpf files in a specified folder.

        Parameters
        ----------
        path : str
            Folder containing files "dc.txt", "gcal.txt", and "program.lpf"
            with information on dot correction, grayscale calibration, and
            a light program file respectively.

        """
        self.load_dc(os.path.join(path, "dc.txt"))
        self.load_gcal(os.path.join(path, "gcal.txt"))
        self.load_lpf(os.path.join(path, "program.lpf"))

    def save_dc(self, file_name):
        """
        Save dc values in a tab-separated text file.

        The resulting file is ready to be used by an LPA.

        Parameters
        ----------
        file_name : str
            Name of the file to save.

        """
        # Flatten dc array
        dc = self.dc.reshape((self.n_rows, self.n_channels*self.n_cols))
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
        """
        Save gcal values in a tab-separated text file.

        The resulting file is ready to be used by an LPA.

        Parameters
        ----------
        file_name : str
            Name of the file to save.

        """
        # Flatten gcal array
        gcal = self.gcal.reshape((self.n_rows, self.n_channels*self.n_cols))
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
        """
        Save grayscale values in a binary .lpf file.

        Grayscale values are calculated from the intensity array using the
        associated LEDSet objects, and the dc and gcal arrays. The
        resulting file is ready to be used by an LPA.

        Parameters
        ----------
        file_name : str
            Name of the file to save.

        """
        n_steps = self.intensity.shape[0]
        # Initialize grayscale array
        gs = numpy.zeros((n_steps,
                          self.n_rows*self.n_cols,
                          self.n_channels), dtype=int)
        # Convert intensities to grayscale values
        for step in range(n_steps):
            for channel in range(self.n_channels):
                try:
                    gs[step, :, channel] = self.led_sets[channel].get_grayscale(
                        row=None,
                        col=None,
                        intensity=self.intensity[step, :, :, channel].flatten(),
                        dc=self.dc[:,:,channel].flatten(),
                        gcal=self.gcal[:,:,channel].flatten(),
                        )
                except ValueError as e:
                    print e.args
                    e.args = ("on step {}, channel {}: ".format(
                        step,
                        channel) + e.args[0],)
                    raise
        # Flatten dimension corresponding to channels
        gs.resize(n_steps, self.n_rows*self.n_cols*self.n_channels)
        # Create LPF object and save
        lpf = LPF()
        lpf.n_channels = self.n_channels*self.n_rows*self.n_cols
        lpf.step_size = self.step_size
        lpf.n_steps = n_steps
        lpf.grayscale = gs
        lpf.save(file_name)

    def save_files(self, path='.'):
        """
        Save dc, gcal, and .lpf files from the contents of this object.

        A folder with the name of this object will be created. Inside,
        files "dc.txt", "gcal.txt", and "program.lpf" will be created.
        An additional empty text file with the name of this object will
        be created.

        Parameters
        ----------
        path : str, optional
            A folder with the name of this object containing all files will
            be created in the directory specified by `path`.

        """
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

    def set_timecourse_staggered(self,
                                 intensity,
                                 intensity_pre,
                                 sampling_steps,
                                 channel,
                                 rows=None,
                                 cols=None):
        """
        Set an intensity timecourse on many wells, with staggered delays.

        The intensity timecourse provided will be displayed on all wells
        indicated by `rows` and `cols`. However, there will be a different
        delay on each well before the intensity timecourse, so that by the
        end of the experiment, well "i" will have seen the intensiy signal
        from ``intensity[0]`` to ``intensity[sampling_steps[i]]``. If the
        samples in the LPA are in steady state (e.g. cells in exponential
        phase), measuring all samples at the end of the experiment is
        equivalent to having one big sample and taking measurements at
        timepoints `sampling_steps[i]*step_size` (given in miliseconds).

        Parameters
        ----------
        intensity : array
            Array with intensity values, one per time step. If
            ``len(intensity)`` is larger than the object's intensity array,
            only the latter part will contain the input `intensity`, and
            the rest will be filled with `intensity_pre`. If the object's
            intensity array is shorter, it will be made large enough to
            hold the `intensity` parameter.
        intensity_pre : float
            Intensity value to use before starting the timecourse.
        sampling_steps : array
            Step numbers at which measurements should be taken.
        channel : int
            LED chanel to use.
        rows, cols : array, optional
            Row and column indices of the wells to use. The length of these
            should be the same as the length of `sampling_steps`.If any of
            these is None, use all wells.

        """
        # Populate row and col arrays if necessary
        if (rows is None) or (cols is None):
            rows = numpy.repeat(numpy.arange(self.n_rows), self.n_cols)
            cols = numpy.tile(numpy.arange(self.n_cols), self.n_rows)
        # Check matching dimensions
        if len(rows) != len(cols):
            raise ValueError("rows and cols should have the same length")
        # Check that the number of sampling steps matches the number of rows and
        # columns
        if len(rows) != len(sampling_steps):
            raise ValueError("number of sampling steps should match the number"\
                " of wells")
        # Expand the intensity array if necessary
        if self.intensity.shape[0] < len(intensity):
            self.set_n_steps(len(intensity))
        # Calculate start of induction step
        n_steps = self.intensity.shape[0]
        start_steps = n_steps - sampling_steps
        # Populate intensity array
        for row, col, start_step in zip(rows, cols, start_steps):
            intensity_well = numpy.ones(n_steps)*intensity_pre
            intensity_well[start_step:] = intensity[:n_steps - start_step]
            self.intensity[:, row, col, channel] = intensity_well

    def discretize_intensity(self):
        """
        Discretize the values in the intensity array.

        At constant dc and gcal values, intensities are represented by
        integer grayscale values ranging from 0 to 4095. Because of this
        finite resolution, only a discrete number of intensity values are
        possible for any given well. This function takes this into account
        to convert the object's intensity values to the closest values that
        are possible at the current dc and gcal values.

        """
        # A separate array will be created and populated. This way, if something
        # goes wrong, we will not overwrite the object's intensity array.
        intensity = numpy.zeros_like(self.intensity)

        for step in range(self.intensity.shape[0]):
            for channel in range(self.n_channels):
                intensity_sch = self.intensity[step, :, :, channel].flatten()
                # Call to LEDSet.discretize_intensity is inside a try block in
                # case the specified intensity is not possible.
                try:
                    intensity_sch = self.led_sets[channel].discretize_intensity(
                        intensity=intensity_sch,
                        dc=self.dc[:,:,channel].flatten(),
                        gcal=self.gcal[:,:,channel].flatten(),
                        )
                except ValueError as e:
                    print e.args
                    e.args = ("on step {}, channel {}: ".format(
                        step,
                        channel) + e.args[0],)
                    raise
                else:
                    intensity_sch.resize(self.n_rows, self.n_cols)
                    intensity[step, :, :, channel] = intensity_sch

        # At this point assume that everything worked, and replace the intensity
        # array
        self.intensity = intensity

    def optimize_dc(self, channel, min_dc=1, uniform=False):
        """
        Get the lowest dc value so that a specified intensity is possible.

        At constant dc and gcal values, intensities are represented by
        integer grayscale values ranging from 0 to 4095. Therefore, at a
        given dc value there is a highest possible intensity. While larger
        dc values allow for larger intensities, resolution is lost, as each
        grayscale step represents a larger increase in intensity as well.
        This function calculates the lowest dc required to achieve the
        maximum intensity in each well, which also maximizes the resolution
        for lower intensities.

        Parameters
        ----------
        channel : int
            Channel on which to optimize dot correction.
        min_dc : array, optional
            Minimum dc value to return.
        uniform : bool, optional
            Whether all returned dc values should be the same.

        """
        # Extract intensities from specified channel
        intensity = self.intensity[:, :, :, channel]
        # Obtain the maximum intensity per well
        intensity_max = intensity.max(axis=0)
        # Obtain optimal dc values
        dc = self.led_sets[channel].optimize_dc(
            intensity=intensity_max.flatten(),
            gcal=self.gcal[:, :, channel].flatten(),
            min_dc=min_dc,
            uniform=uniform)
        # Resize and replace current dc values
        dc.resize(self.n_rows, self.n_cols)
        self.dc[:, :, channel] = dc

    def plot_intensity(self,
                       channel,
                       file_name=None,
                       xunits='s',
                       ylim=(0.1, 200),
                       yscale='log',
                       figsize=(12, 8)):
        """
        Plot the light intensity for each well in the LPA.

        This function creates a ``n_rows x n_cols`` array of subplots, each
        one of which will contain the intensities in time of each well, for
        the specified channel.

        Parameters
        ----------
        channel : int
            Channel from which to make the plot.
        file_name : str, optional
            Name of the file to save the figure to. If None, don't save.
        xunits : {'step', 'ms', 's', 'min'}, optional
            Units to use for the x axis. This can be either the raw step
            number, miliseconds, seconds, or minutes.
        ylim : tuple, optional
            Limit for the y axis.
        yscale : {'linear', 'log'}, optional
            Scale for the y axis.
        figsize : tuple, optional
            Size of the figure to make.

        """
        pyplot.figure(figsize=figsize)
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                # Subplot position should match well position in plate
                pyplot.subplot(self.n_rows,
                               self.n_cols,
                               row*self.n_cols + col + 1)
                # Calculate x axis data based on units
                if xunits=='step':
                    time = numpy.arange(self.intensity.shape[0])
                elif xunits=='ms':
                    time = numpy.arange(self.intensity.shape[0]) * \
                        self.step_size
                elif xunits=='s':
                    time = numpy.arange(self.intensity.shape[0]) * \
                        self.step_size/1000.
                elif xunits=='min':
                    time = numpy.arange(self.intensity.shape[0]) * \
                        self.step_size/60000.
                else:
                    raise ValueError('units for x axis not recognized')
                # Plot
                pyplot.step(time,
                            self.intensity[:, row, col, channel])
                # Set labels, scales, lims
                pyplot.xlim(time[0], time[-1])
                pyplot.xlabel('Time ({})'.format(xunits))
                pyplot.ylim(ylim)
                pyplot.yscale(yscale)
                pyplot.ylabel('Intensity (umol/m2/s)')

        # Save if necessary
        if file_name is not None:
            pyplot.tight_layout()
            pyplot.savefig(file_name, dpi=200)
            pyplot.close()
