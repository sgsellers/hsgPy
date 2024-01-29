import configparser
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.signal as ssig

class hsgCal:
    """
    The Sunspot Solar Observatory Consortium's software for reducing
    HSG (Horizontal Steerable Grating) data from the Dunn Solar Telescope.
    Note that this pipeline assumes that HSG data were taken using the AIW
    interface as the DST, and not operated through the SPINOR instrument GUI.

    ------------------------------------------------------------------------

    Use this software to process/reduce data from the HSG instrument at DST.
    To perform reductions using this package:
    1.) Install this package, along with the rest of SSOSoft.
    2.) Set the necessary instrument parameters in a configuration file.
        (Use the included sampleConfig.ini as a template)
    3.) Open a Python/iPython terminal and "from hsgCal import hsgCal"
    4.) Start an instance of the calibration class by using
        'h=hsgCal('<CAMERA>', '<CONFIGFILE>')'
    5.) Run the standard calibration method using
        'h.hsg_run_calibration()'

    Parameters
    ----------
    camera : str
        A string containing the camera name.
        Common values will be
            -SARNOFF_5876
            -SARNOFF_5896
            -SARNOFF_8542
            -FLIR_6302
            -SI_805_6302
            -PCO_6563
    configFile : str
        Path to the configuration file

    To my knowledge, a complete calibration should do the following:
        1.) Create average solar flat, lamp flat, and dark files
        2.) Use the solar flat to determine beam size and hairline positions, then clip the images to these values.
        3.) Derotate and deskew the dark-corrected flat.
            a.) Deskew and derotate are done by the same function.
                Deskew uses a spectral line.
                Derotate uses a hairline and is rotated to accomodate the function
        4.) Create a gaintable using the deskew to iteratively detrend the spectral profiles from the flat field
        5.) Perform wavelength calibration with deskewed flat field against FTS atlas, determine wavelength array.
        6.) Read in the data files, and perform the following corrections:
            a.) Apply derotation and deskew shifts from flat fields.
            b.) Align hairlines with flat hairlines
            c.) Subtract dark
            d.) Divide by lampgain if available
            e.) Divide by solar gain
        7.) Attempt a prefilter correction
        8.) Attempt a fringe correction
            Note that 7 & 8 might be easier to swap in order, since the fringes, particularly in the Sarnoff cameras
            at 8542A, affect the spectral shape pretty severely.
        9.) Write fits files for each raster.
            These files should have a master header in slot 0 with no data corresponding
            Then extension 1 has the datacube with data-specific parameters. Datacube is nx, ny, nlambda
            Extension 2 has the wavelength array with dimensions nlambda
    """

    def __init_(self, camera, configFile):
        """
        Parameters:
        -----------
        camera : str
            String containing the camera name.
        configFile : str
            Path to the configuration file
        """

        try:
            f = open(configFile, 'r')
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.configFile = configFile
        self.camera = camera.upper()

        self.avgDark = None
        self.solarFlat = None
        self.lampFlat = None

        self.darkList = [""]
        self.solarFlatList = [""]
        self.lampFlatList = [""]

        self.dataList = [""]
        self.dataShape = None
        self.dataBounds = None

        self.skewShifts = None


    def hsg_average_image_from_list(self, fileList):
        """
        Computes an average image from a list of HSG image files.

        Parameters:
        -----------
        fileList : list
            A list of file paths to the images to be averaged

        Returns:
        --------
        numpy.ndarray
            2-Dimensional with dtype float
        """
        if len(fileList) == 0:
            return None
        testImg = fits.open(fileList[0])
        testExt = testImg[1].data
        # TCOPS makes spectral images in the shape (1, NY, NX)
        avgImg = np.zeros(testExt[0].shape)
        imNum = 0
        for file in fileList:
            with fits.open(file) as hdu:
                for ext in hdu[1:]:
                    avgImg += ext.data[0]
                    imNum += 1
        avgImg = avgImg / imNum
        return avgImg

"""
"""