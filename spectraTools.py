import matplotlib

import numpy as np
import numpy.polynomial.polynomial as npoly
import matplotlib.pyplot as plt
import scipy.interpolate as scinterp
import scipy.ndimage as scind
import scipy.optimize as scopt
import scipy.signal as scig
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


def select_lines_singlepanel_unbound(array):
    """
    Matplotlib-based function to select an x-value, or series of x-values
    From the plot of a 1D array.

    :param array: array-like
        Array to plot and select from
    :return xvals: array-like
        Array of selected x-values with length nselections
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Select Positions, then close window")
    spectrum, = ax.plot(array)

    xvals = []

    def onselect(event):
        xcd = event.xdata
        ax.axvline(xcd, c='C1', linestyle=':')
        fig.canvas.draw()
        xvals.append(xcd)
        print("Selected: " + str(xcd))

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


def detect_beams_hairlines(image, threshold=0.5, hairline_width=5, line_width=15):
    """
    Detects beam edges and intensity thresholds from an image (typically a flat).
    This function makes no assumptions as to the number of beams/slits used.
    Theoretically, it should work on a quad slit dual beam configuration from FIRS,
    which to my knowledge has not been used in some time.
    
    It works on derivative intensity thresholds. First across dimension 1 (left/right), detecting the number of slits
    from the averaged profile. Then across dimansion 2 (up/down) for each slit. This detects the top/bottom of the beam
    (important for FIRS and other polarizing beam split instruments), as well as smaller jumps from the harlines.
    
    :param image: numpy.ndarray
        2D image (typically an averaged flat field) for beam detection.
    :param threshold: float
        Threshold for derivative profile separation.
    :param hairline_width: int
        Maximum width of hairlines in pixels
    :param line_width: int
        Maximum width of spectal line in pixels
    :return beam_edges: numpy.ndarray
        Numpy.ndarray of shape (2, nBeams), where each pair is (y0, y1) for each beam
    :return slit_edges: numpy.ndarray
        Numpy.ndarray of shape (2, nSlits), where each pair is (x0, x1) for each slit
    :return hairline_centers: numpy.ndarray
        Locations of hairlines. Shape is 1D, with length nhairlines, and the entries are the center of the hairline
    """
    mask = image.copy()
    thresh_val = threshold * np.nanmedian(mask)
    mask[mask <= thresh_val] = 1
    mask[mask >= thresh_val] = 0

    # Hairlines and y-limits first
    yprofile = np.nanmedian(mask, axis=1)
    yprofile_grad = np.gradient(yprofile)
    pos_peaks, _ = scig.find_peaks(yprofile_grad)
    neg_peaks, _ = scig.find_peaks(-yprofile_grad)
    # Every peak that has no corresponding opposite sign peak within hairline width
    # is a beam edge. Otherwise, it's a hairline.
    hairline_starts = []
    edges = []
    for peak in pos_peaks:
        if len(neg_peaks[(neg_peaks >= peak - hairline_width) & (neg_peaks <= peak + hairline_width)]) > 0:
            hairline_starts.append(peak)
        else:
            edges.append(peak)
    hairline_ends = []
    for peak in neg_peaks:
        if len(pos_peaks[(pos_peaks >= peak - hairline_width) & (pos_peaks <= peak + hairline_width)]) > 0:
            hairline_ends.append(peak)
        else:
            edges.append(peak)
    # Sort the beam edges
    # Should also pad with 0 and -1 if there's no gap between the beam and the detector edge
    # We'll do this by checking if the first edge is < 100, and adding 0 as the first edge if it isn't
    # Then, if the length of edges is even, we're good. If it's odd, we add the last index as well.
    edges = sorted(edges)
    if edges[0] > 100:
        edges = [0] + edges
    if len(edges) % 2 == 1:
        edges.append(len(yprofile)-1)
    # Now we'll check the hairlines, and return the mean value of the start/end pair
    hairline_centers = []
    for i in range(len(hairline_starts)):
        hairline_centers.append((hairline_starts[i] + hairline_ends[i])/2)

    # We can use similar logic to find the beam edges in x.
    # Flatten in the other direction, and avoid spectral line residuals in the same way we picked out hairlines
    # We'll use a different, wider window for the spectral line avoidance, specifically for Ca II 8542 and H alpha
    # 10 should be okay.
    xprofile = np.nanmedian(mask, axis=0)
    xprofile_grad = np.gradient(xprofile)
    pos_peaks, _ = scig.find_peaks(xprofile_grad)
    neg_peaks, _ = scig.find_peaks(-xprofile_grad)
    xedges = []
    for peak in pos_peaks:
        if len(neg_peaks[(neg_peaks >= peak - line_width) & (neg_peaks <= peak + line_width)]) == 0:
            xedges.append(peak)
    for peak in neg_peaks:
        if len(pos_peaks[(pos_peaks >= peak - line_width) & (pos_peaks <= peak + line_width)]) == 0:
            xedges.append(peak)
    # Clean it up. If there's no edges found, the beam fills the chip, index to 0, -1
    if len(xedges) == 0:
        xedges = [0, len(xprofile)-1]
    # If there's no edges found in the first 50, add a zero to the front
    xedges = sorted(xedges)
    if xedges[0] > 50:
        xedges = [0] + xedges
    # If there are now an even number of edges, most likely situation is that we missed the end of the last slit.
    if len(xedges) % 2 == 1:
        xedges.append(len(xprofile)-1)

    beam_edges = np.array(edges).reshape(int(len(edges)/2), 2)
    slit_edges = np.array(xedges).reshape(int(len(xedges)/2), 2)
    hairline_centers = np.array(hairline_centers)
    return beam_edges, slit_edges, hairline_centers


def create_gaintables(flat, lines_indices,
                      hairline_positions=None, neighborhood=50,
                      iterations=3, hairline_width=3, edge_padding=3):
    """
    Creates the gain of a given input flat field beam.
    This assumes a deskewed field, and will determine the shift of the template mean profile, then detrend the spectral
    profile, leaving only the background detector flat field.
    :param flat: numpy.ndarray
        Dark-corrected flat field image to determine the gain for
    :param lines_indices: list
        Indices of the spectral line to use for deskew. Form [idx_low, idx_hi]
    :param hairline_positions: list
        List of hairline y-centers. If None, hairlines are not masked. May cause issues in line position finding.
    :param neighborhood: int
        Size of region to use for mean profile calculation.
        Selects a region at the position of the profile +/- neighborhood/2
    :param iterations: int
        Number of iterations to perform on the gain table.
    :param hairline_width: int
        Width of hairlines for masking. Default is 3
    :param edge_padding: int
        Cuts the profile arrays by this amount on each end to avoid edge effects.
    :return gaintable: numpy.ndarray
        Gain table from iterating along overlapping subdivisions
    :return coarse_gaintable: numpy.ndarray
        Gain table from using full slit-averaged profile
    :return init_skew_shifts: numpy.ndarray
        Shifts used in initial flat field deskew. Can be applied to final gain-corrected science maps.
    """
    masked_flat = flat.copy()
    if hairline_positions is not None:
        for line in hairline_positions:
            masked_flat[int(line - hairline_width - 1):int(line + hairline_width), :] = np.nan
    init_skew_shifts = spectral_skew(masked_flat[:, lines_indices[0]:lines_indices[1]])
    init_deskew = np.zeros(masked_flat.shape)
    for i in range(masked_flat.shape[0]):
        init_deskew[i, :] = scind.shift(masked_flat[i, :], init_skew_shifts[i], mode='nearest')
    mean_profile = np.nanmean(
        init_deskew[
            int(init_deskew.shape[0]/2 - 30):int(init_deskew.shape[0]/2 + 30), :
        ],
        axis=0
    )
    mean_profile_center = find_line_core(mean_profile[lines_indices[0]-3:lines_indices[1]+3]) + lines_indices[0] - 3
    shifted_lines = np.zeros(masked_flat.shape)
    sh = []
    for i in range(masked_flat.shape[0]):
        if i == 0:
            last_nonnan = np.nan
        line_position = find_line_core(
            masked_flat[i, lines_indices[0]-3:lines_indices[1]+3]
        ) + lines_indices[0] - 3
        if np.isnan(line_position):
            if np.isnan(last_nonnan):
                shift = 0
            else:
                shift = last_nonnan - mean_profile_center
        else:
            shift = line_position - mean_profile_center
            last_nonnan = line_position
        sh.append(shift)
        shifted_lines[i, :] = scind.shift(mean_profile, shift, mode='nearest')
    coarse_gaintable = flat / shifted_lines
    if hairline_positions is not None:
        for line in hairline_positions:
            coarse_gaintable[int(line - hairline_width - 1):int(line + hairline_width), :] = 1

    # Smooth rough gaintable in the chosen line
    if lines_indices[0] < 20:
        lowidx = 0
    else:
        lowidx = lines_indices[0] - 20
    if lines_indices[1] > flat.shape[0] - 20:
        highidx = flat.shape[0] - 1
    else:
        highidx = lines_indices[1] + 20
    for i in range(coarse_gaintable.shape[1]):
        coarse_gaintable[lines_indices[0] - 7:lines_indices[1] + 7, i] = np.nanmean(coarse_gaintable[lowidx:highidx, i])

    corrected_flat = masked_flat / coarse_gaintable

    master_shifts = np.zeros((iterations, corrected_flat.shape[0]))
    for i in range(iterations):
        skew_shifts = spectral_skew(corrected_flat[:, lines_indices[0]:lines_indices[1]])
        deskew_corrected_flat = np.zeros(corrected_flat.shape)
        for j in range(corrected_flat.shape[0]):
            deskew_corrected_flat[j, :] = scind.shift(corrected_flat[j, :], skew_shifts[j], mode='nearest')
        shifted_lines = np.zeros(corrected_flat.shape)
        for j in range(corrected_flat.shape[0]):
            if np.isnan(corrected_flat[j, 0]):
                continue
            if j < int(neighborhood/2):
                mean_profile = np.nanmean(deskew_corrected_flat[:neighborhood, :], axis=0)
            elif j > corrected_flat.shape[0] - int(neighborhood/2):
                mean_profile = np.nanmean(deskew_corrected_flat[-neighborhood:, :], axis=0)
            else:
                mean_profile = np.nanmean(deskew_corrected_flat[j-int(neighborhood/2):j+int(neighborhood/2), :], axis=0)
            mean_profile = scind.shift(mean_profile, -skew_shifts[j], mode='nearest')
            line_shift = scopt.minimize_scalar(
                fit_profile,
                bounds=(-1, 1),
                args=(
                    corrected_flat[j, edge_padding:-edge_padding],
                    mean_profile[edge_padding:-edge_padding]
                )
            ).x
            shifted_lines[j, :] = scind.shift(mean_profile, line_shift, mode='nearest')
        gaintable = flat/shifted_lines
        if hairline_positions is not None:
            for line in hairline_positions:
                gaintable[int(line - hairline_width - 1):int(line + hairline_width), :] = 1
        corrected_flat = masked_flat / gaintable

    return gaintable, coarse_gaintable, init_skew_shifts


def fit_profile(shift, reference_profile, mean_profile):
    """
    For use with scipy.optimize.curve_fit; this function shifts reference profile by "shift"
    and returns its chi-squared value relative to mean_profile

    :param shift: float
        Value for shift
    :param reference_profile: array-like
        The reference profile to shift against
    :param mean_profile: array-like
        The mean profile to shift to match reference
    :return chisq: float
        Chisquared value for reference relative to shifted mean profile
    """
    shifted_mean = scind.shift(mean_profile, shift, mode='nearest')
    return chi_square(shifted_mean, reference_profile)


def chi_square(fit, prior):
    return np.nansum((fit - prior)**2)/len(fit)
