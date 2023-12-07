import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
import numpy.polynomial.polynomial as npoly
import matplotlib.pyplot as plt
import scipy.interpolate as scinterp
import scipy.ndimage as scind
import warnings

from importlib import resources
import FTS_atlas

"""
This file contains generalized helper functions for Spectrograph/Spectropolarimeter
reductions. While the actual reduction packages are specific to the instrument, these
functions represent bits of code that can be reused at a low level. See the individual
docstrings for a comprehensive explanation of each function given below.
"""

def find_nearest(array, value):
    """
    Determines the index of the closest value in the array to the provided value
    :param array: array-like
        array
    :param value: float
        value
    :return idx: int
        index
    """
    array = np.nan_to_num(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def find_line_core(profile, wvl=None):
    """
    Uses the Fourier Phase method to determine the position of a spectral line core.
    See Schlichenmaier & Schmidt, 2000 for details on this application.
    It's fast, insensitive to noise, but does require a very narrow range.

    :param profile: array-like
        The line profile to determine the centroid of. Assumed to be 1-D
    :param wvl: array-like
        Optional, the wavelengths corresponding to the line profile.
        If wvl is given, the returned value will be a wavelength
    :return center: float
        The position of the line core
        If wvl is not given, in pixel number
        Otherwise, in wavelength space.
    """

    profile_fft = np.fft.fft(profile)
    center = -np.arctan(
        np.imag(profile_fft[1])/np.real(profile_fft[1])
    ) / (2*np.pi) * len(profile) + (len(profile)/2.)
    if wvl is not None:
        center = scinterp.interp1d(np.arange(len(wvl)), wvl, kind='linear')(center)
    return center


def fts_window(wavemin, wavemax, atlas='FTS', norm=True, lines=False):
    """
    For a given wavelength range, return the solar reference spectrum within that range.

    :param wavemin: float
        Blue end of the wavelength range
    :param wavemax: float
        Red end of the wavelength range
    :param atlas: str
        Which atlas to use. Currently accepts "Wallace" and "FTS"
        Wallace uses the 2011 Wallace updated atlas
        FTS uses the 1984 FTS atlas
    :param norm: bool
        If False, and the atlas is set to "FTS", will return the solar irradiance.
        This includes the blackbody curve, etc.
    :param lines: bool
        If True, returns additional arrays denoting line centers and names
        within the wavelength range.
    :return wave: array-like
        Array of wavelengths
    :return spec: array-like
        Array of spectral values
    :return line_centers: array-like, optional
        Array of line center positions
    :return line_names: array-like, optional
        Array of line names
    """

    def read_data(path, fname) -> np.array:
        with resources.path(path, fname) as df:
            return np.load(df)

    if atlas.lower() == 'wallace':
        if (wavemax <= 5000.) or (wavemin <= 5000.):
            atlas_angstroms = read_data('FTS_atlas', 'Wallace2011_290-1000nm_Wavelengths.npy')
            atlas_spectrum = read_data('FTS_atlas', 'Wallace2011_290-1000nm_Observed.npy')
        else:
            atlas_angstroms = read_data('FTS_atlas', 'Wallace2011_500-1000nm_Wavelengths.npy')
            atlas_spectrum = read_data('FTS_atlas', 'Wallace2011_500-1000nm_Corrected.npy')
    else:
        atlas_angstroms = read_data('FTS_atlas', 'FTS1984_296-1300nm_Wavelengths.npy')
        if norm:
            atlas_spectrum = read_data('FTS_atlas', 'FTS1984_296-1300nm_Atlas.npy')
        else:
            warnings.warn("Using solar irradiance (i.e., not normalized)")
            atlas_spectrum = read_data('FTS_atlas', 'FTS1984_296.-1300nm_Irradiance.npy')

    idx_lo = find_nearest(atlas_angstroms, wavemin) - 5
    idx_hi = find_nearest(atlas_angstroms, wavemax) + 5

    wave = atlas_angstroms[idx_lo:idx_hi]
    spec = atlas_spectrum[idx_lo:idx_hi]

    if lines:
        line_centers_full = read_data(
            'FTS_atlas',
            'RevisedMultiplet_Linelist_2950-13200_CentralWavelengths.npy'
        )
        line_names_full = read_data(
            'FTS_atlas',
            'RevisedMultiplet_Linelist_2950-13200_IonNames.npy'
        )
        line_selection = (line_centers_full < wavemax) & (line_centers_full > wavemin)
        line_centers = line_centers_full[line_selection]
        line_names = line_names_full[line_selection]
        return wave, spec, line_centers, line_names
    else:
        return wave, spec


def rolling_median(data, window):
    """
    Simple rolling median function, rolling by the central value.
    Preserves the edges to provide an output array of the same shape as the input.
    I wasn't a fan of any of the prewritten rolling median function edge behaviours.
    Hence this. Kind of a kludge, tbh.

    :param data: array-like
        Array of data to smooth
    :param window: int
        Size of median window.
    :return rolled: array-like
        Rolling median of input array
    """

    rolled = np.zeros(len(data))
    half_window = int(window/2)
    if half_window >= 4:
        for i in range(half_window):
            rolled[i] = np.nanmedian(data[i:i+1])
            rolled[-(i+1)] = np.nanmedian(data[(-(i+4)):(-(i+1))])
    else:
        rolled[:half_window] = data[:half_window]
        rolled[-(half_window + 1):] = data[-(half_window + 1):]
    for i in range(len(data) - window):
        rolled[half_window + i] = np.nanmedian(data[i:half_window + i])
    return rolled


def select_lines_singlepanel(array, nselections):
    """
    Matplotlib-based function to select an x-value, or series of x-values
    From the plot of a 1D array.

    :param array: array-like
        Array to plot and select from
    :param nselections: int
        Number of expected selections
    :return xvals: array-like
        Array of selected x-values with length nselections
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Select " + str(nselections) + " Positions, then Click Again to Exit")
    spectrum, = ax.plot(array)

    xvals = []
    def onselect(event):
        if len(xvals) < nselections:
            xcd = event.xdata
            ax.axvline(xcd, c='C1', linestyle=':')
            fig.canvas.draw()
            xvals.append(xcd)
            print("Selected: " + str(xcd))
        else:
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect)
    plt.show()
    xvals = np.array(xvals)
    return xvals


def select_lines_doublepanel(array1, array2, nselections):
    """
    Matplotlib-based function to select an x-value, or series of x-values
    From two plots of 1D arrays.

    :param array1: array-like
        First array to plot and select from
    :param array2: array-like
        Second array to plot and select from.
    :param nselections: int
        Number of expected selections
        NOTE: It is assumed that nselections are split evenly between
        array1 and array2. Please don't try to break this.
    :return xvals1: array-like
        Array of selected x-values from array1 with length nselections/2
    :return xvals2: array-like
        Array of selected x-values from array2 with length nselections/2
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.suptitle("Select " + str(int(nselections/2)) + " Positions on each plot")
    spectrum1, = ax1.plot(array1)
    spectrum2, = ax2.plot(array2)

    xvals1 = []
    xvals2 = []
    def onselect(event):
        if event.inaxes == ax1:
            if len(xvals1) < int(nselections/2):
                xcd = event.xdata
                ax1.axvline(xcd, c='C1', linestyle=':')
                fig.canvas.draw()
                xvals1.append(xcd)
                print("Selected: " + str(xcd))
        elif event.inaxes == ax2:
            if len(xvals2) < int(nselections/2):
                xcd = event.xdata
                ax2.axvline(xcd, c='C1', linestyle=':')
                fig.canvas.draw()
                xvals2.append(xcd)
                print("Selected: " + str(xcd))

        if (len(xvals1) >= int(nselections/2)) & (len(xvals2) >= int(nselections/2)):
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect)
    plt.show()
    xvals1 = np.array(xvals1)
    xvals2 = np.array(xvals2)
    return xvals1, xvals2


def spectral_skew(image, order=2, slit_reference=0.5):
    """
    Adaptation of the deskew1.pro function included in firs-soft, spinor-soft.
    In the y-direction of the input array, it determines the position of the line
    core, then fits a polynomial of "order" (default 2) to these shifts, as well as
    a line. It then normalizes the shifts relative to the position of the linear fit
    at "slit_reference" along the slit. It then returns these shifts for use with
    scipy.ndimage.shift.

    Ideally, you'd give this function an image that's narrow in wavelength space.
    Possibly several times, then average the relative shifts, or create a profile
    of shifts in the wavelength direction of the wider image.
    :param image: array-like
        2D array of data to find offsets to. Assumes the slit is in the y-direction,
        with wavelength in the x-direction. Also assumes that the hairlines are masked
        out as NaN.
    :param order: int
        Order of polynomial to determine the relative shift profile along the slit
    :param slit_reference: float
        Fractional height of the image to determine shifts relative to. Default in
        deskew1.pro was 0.25. My quick tests showed 0.5 working better. It's a keyword arg.
    :return shifts: array-like
        1D array of shifts along the slit for the provided line core.
    """

    core_positions = np.zeros(image.shape[0])
    for i in range(len(core_positions)):
        core_positions[i] = find_line_core(image[i, :])

    # We assume that the hairlines are NaN slices.
    # These return NaN from find_line_core
    # We need to cut the NaN values, while preserving the spacing along y.
    yrange = np.arange(image.shape[0])
    nancut = np.nan_to_num(core_positions) != 0
    core_positions_tofit = core_positions[nancut]
    yrange_tofit = yrange[nancut]

    # I very earnestly miss when it was just numpy.polyfit
    # And not numpy.polynomial.polynomial.Polynomial.fit().convert().coef
    # It feels like they're playing a joke on me.
    polycoeff = npoly.Polynomial.fit(yrange_tofit, core_positions_tofit, order).convert().coef
    lincoeff = npoly.Polynomial.fit(yrange_tofit, core_positions_tofit, 1).convert().coef

    core_polynomial = np.zeros(image.shape[0])
    for i in range(len(polycoeff)):
        core_polynomial += polycoeff[i] * yrange**i
    core_linear = lincoeff[0] + yrange*lincoeff[1]

    shifts = -(core_polynomial - core_linear[int(slit_reference * image.shape[0])])

    if np.nanmean(shifts) >= 7.5:
        warnings.warn("Large average shift ("+str(np.nanmean(shifts))+") measured along the slit. Check your inputs.")

    return shifts

