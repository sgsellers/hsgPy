import astropy.io.fits as fits
import configparser
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as scind
import scipy.interpolate as scinterp
import tqdm
from astropy.constants import c
c_kms = c.value/1e3
from . import spectraTools as spex


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
        3.) Deskew the dark-corrected flat.
        4.) Create a gaintable using the deskew to iteratively detrend the spectral profiles from the flat field
        5.) Perform wavelength calibration with deskewed flat field against FTS atlas, determine wavelength array.
        6.) Read in the data files, and perform the following corrections:
            a.) Subtract dark
            b.) Divide by lampgain if available
            c.) Divide by solar gain
            d.) Apply deskew shifts from flat fields.
            e.) Align hairlines with flat hairlines
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
        self.numDark = 0
        self.solarFlat = None
        self.numFlat = 0
        self.lampGain = None
        self.numLamp = 0
        self.solarGain = None
        self.coarseGain = None
        # Required for a few things, might as well store it.
        self.deskewedFlat = None

        self.baseDir = ""
        self.reduceDir = ""
        self.finalDir = ""

        self.darkList = [""]
        self.solarFlatList = [""]
        self.lampFlatList = [""]
        self.scanList = [""]

        self.darkFile = ""
        self.solarFlatFile = ""
        self.solarGainFile = ""
        self.lampGainFile = ""

        self.dataList = [""]
        self.dataShape = None
        self.dataBounds = None

        self.skewShifts = None
        self.beamEdges = []
        self.spectralEdges = []
        self.hairlines = []

        self.lineSelectionMethod = ""
        self.lineSelectionIndices = []
        self.ftsSelectionIndices = []
        self.deskewShifts = []
        self.fringeMethod = ""

        self.centralWavelength = 0
        self.approxDispersion = 0
        self.slitWidth = 0
        self.stepSpacing = 0
        self.arcsecPerPixel = 0
        self.beamThreshold = 0.5

        self.referenceFTSspec = []
        self.referenceFTSwave = []
        self.wavelengthArray = []

        self.velocityMapLineIndex = None


    def hsg_run_calibration(self):
        """
        The main calibration method for standard HSG data.
        """
        self.hsg_configure_run()
        self.hsg_get_cal_images()  # Includes creating cal images if req'd
        self.hsg_save_cal_images()
        if self.fringeMethod != "NONE":
            self.hsg_fringe_correction()
        self.hsg_wavelength_calibration()
        self.hsg_perform_scan_calibration(selectLine=True)  # Includes velocity determination of selected line
        return


    def hsg_configure_run(self):
        """
        Reads configuration file and sets up parameters necessary for calibration run.
        """
        def hsg_assert_file_list(flist):
            assert (len(flist) != 0), "List contains no matches."

        config = configparser.ConfigParser()
        config.read(self.configFile)

        self.baseDir = config[self.camera]['baseDir']
        self.reduceDir = config[self.camera]['reduceDir']
        self.finalDir = os.path.join(self.reduceDir, "calibratedScans")
        if not os.path.isdir(self.finalDir):
            print("{0}: os.mkdir: attempting to create directory:"
                  "{1}".format(__name__, self.finalDir))
            try:
                os.mkdir(self.finalDir)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise

        self.darkFile = os.path.join(self.reduceDir, "{0}_DARK.fits".format(self.camera))
        self.solarFlatFile = os.path.join(self.reduceDir, "{0}_SOLARFLAT.fits".format(self.camera))
        self.solarGainFile = os.path.join(self.reduceDir, "{0}_SOLARGAIN.fits".format(self.camera))
        self.lampGainFile = os.path.join(self.reduceDir, "{0}_LAMPGAIN.fits".format(self.camera))

        self.solarFlatList = sorted(glob.glob(os.path.join(
            self.baseDir, config[self.camera]['solarFlatPattern']
        )))
        try:
            hsg_assert_file_list(self.solarFlatList)
        except AssertionError as err:
            print("Error: solarFlatList: {0}".format(err))
            raise
        else:
            print("Files in solar flat list: {0}".format(len(self.solarFlatList)))

        self.darkList = sorted(glob.glob(os.path.join(
            self.baseDir, config[self.camera]['darkPattern']
        )))
        try:
            hsg_assert_file_list(self.darkList)
        except AssertionError as err:
            print("Error: darkList: {0}".format(err))
            raise
        else:
            print("Files in dark list: {0}".format(len(self.darkList)))
        if 'lampFlatPattern'.lower() in list(config[self.camera].keys()):
            self.lampFlatList = sorted(glob.glob(os.path.join(
                self.baseDir, config[self.camera]['lampFlatPattern']
            )))
            try:
                hsg_assert_file_list(self.darkList)
            except AssertionError as err:
                print("Error: lampFlatList: {0}".format(err))
                pass
            else:
                print("Files in lamp flat list: {0}".format(len(self.lampFlatList)))

        self.scanList = sorted(glob.glob(os.path.join(
            self.baseDir, config[self.camera]['dataPattern']
        )))
        try:
            hsg_assert_file_list(self.scanList)
        except AssertionError as err:
            print("Error: scanList: {0}".format(err))
            raise
        else:
            print("Files in scan list: {0}".format(len(self.scanList)))

        self.centralWavelength = float(config[self.camera]['centralWavelength'])
        self.approxDispersion = float(config[self.camera]['approximateDispersion'])
        self.slitWidth = float(config[self.camera]['slitwidth'])
        self.arcsecPerPixel = float(config[self.camera]['arcsecPerPixelY'])
        if 'beamEdgeThreshold'.lower() in list(config[self.camera].keys()):
            self.beamThreshold = float(config[self.camera]['beamEdgeThreshold'])

        if len(config[self.camera]['lineSelection'].split(",")) > 1:
            self.lineSelectionIndices = [
                int(x) for x in config[self.camera]['lineSelection'].split(',')
            ]
            self.lineSelectionMethod = "USER"
        else:
            self.lineSelectionMethod = "MANUAL"

        self.fringeMethod = config[self.camera]['fringeCorrection'].upper()


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
            return None, 0
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
        return avgImg, imNum


    def hsg_get_cal_images(self):
        """
        If the average dark, solar flat, lamp gain, and solar gain files exist, reads them into the class.
        Otherwise, creates these files and saves them.
        """
        if os.path.exists(self.darkFile):
            with fits.open(self.darkFile) as hdu:
                self.avgDark = hdu[0].data
        else:
            self.avgDark, self.numDark = self.hsg_average_image_from_list(self.darkList)
        if os.path.exists(self.solarFlatFile):
            with fits.open(self.solarFlatFile) as hdu:
                self.avgFlat = hdu[0].data
        else:
            self.avgFlat, self.numFlat = self.hsg_average_image_from_list(self.solarFlatList)
        if os.path.exists(self.lampGainFile):
            with fits.open(self.lampGainFile) as hdu:
                self.lampGain = hdu[0].data
        else:
            self.lampGain, self.numLamp = self.hsg_compute_lamp_gain()
        if os.path.exists(self.solarGainFile):
            with fits.open(self.solarGainFile) as hdu:
                self.solarGain = hdu[0].data
                self.coarseGain = hdu[1].data
                self.beamEdges = [hdu[0].header['BEAM1'], hdu[0].header['BEAM2']]
                self.spectralEdges = [hdu[0].header['SLIT1'], hdu[0].header['SLIT2']]
                hairs = [key for key in list(hdu[0].header.keys()) if "HAIR" in key]
                self.hairlines = [hdu[0].header[key] for key in hairs]
        else:
            self.hsg_compute_gain()
        return


    def hsg_save_cal_images(self):
        """
        Save Flat, Dark, Lampgain, and Gain Files. I'm choosing not to write the FITS writer function general
        enough to write both cals and data files, as the header requirements are more complicated for the data
        files, and I still want SOME information in the cal headers. Writing a function to do both would necessitate
        another function to contruct cal file specific fits headers, which, while best practice,
        is not neccessary for our use case.
        :return:
        """
        if not os.path.exists(self.darkFile):
            hdu = fits.HDUList([fits.PrimaryHDU(self.avgDark)])
            hdu[0].header['DATATYPE'] = "AVGDARK"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numDark
            hdu.writeto(self.darkFile, overwrite=True)
        if not os.path.exists(self.solarFlatFile):
            hdu = fits.HDUList([fits.PrimaryHDU(self.avgFlat)])
            hdu[0].header['DATATYPE'] = "AVGSFLAT"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numFlat
            hdu.writeto(self.solarFlatFile, overwrite=True)
        if not os.path.exists(self.lampGainFile) & self.lampGain:
            hdu = fits.HDUList([fits.PrimaryHDU(self.lampGain)])
            hdu[0].header['DATATYPE'] = "LAMPGAIN"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numLamp
            hdu.writeto(self.lampGainFile, overwrite=True)
        if not os.path.exists(self.solarGainFile):
            hdu = fits.HDUList([fits.PrimaryHDU(self.solarGain), fits.PrimaryHDU(self.coarseGain)])
            hdu[0].header['DATATYPE'] = "SGAIN"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numFlat
            hdu[0].header['BEAM1'] = self.beamEdges[0]
            hdu[0].header['BEAM2'] = self.beamEdges[1]
            hdu[0].header['SLIT1'] = self.spectralEdges[0]
            hdu[0].header['SLIT2'] = self.spectralEdges[1]
            for i in range(len(self.hairlines)):
                hdu[0].header['HAIR'+str(i+1)] = self.hairlines[i]
            hdu.writeto(self.solarGainFile, overwrite=True)
        return


    def hsg_compute_lamp_gain(self):
        """
        Creates average lamp gain image (that is, dark corrected and normalized)
        """
        avgLampFlat, numLamp = self.hsg_average_image_from_list(self.lampFlatList)
        if avgLampFlat:
            lampGain = avgLampFlat - self.avgDark
            lampGain = lampGain / np.nanmedian(lampGain)
            return lampGain, numLamp
        else:
            return None, 0


    def hsg_compute_gain(self):
        """
        Computes solar gain and finds the beam edges
        :return:
        """
        # Step 0: Determine beam/spectral edges & hairline positions.
        beam_edges, slit_edges, hairlines = spex.detect_beams_hairlines(
            self.avgFlat,
            threshold=self.beamThreshold
        )
        self.beamEdges = beam_edges[0]
        self.spectralEdges = slit_edges[0]
        # Position of hairlines AFTER image is cropped.
        self.hairlines = hairlines - self.beamEdges[0]

        if self.lampGain:
            croppedFlat = (self.avgFlat - self.avgDark) / self.lampGain
            croppedFlat = croppedFlat[
                self.beamEdges[0]:self.beamEdges[1],
                self.spectralEdges[0]:self.spectralEdges[1]
            ]
        else:
            croppedFlat = self.avgFlat - self.avgDark
            croppedFlat = croppedFlat[
                self.beamEdges[0]:self.beamEdges[1],
                self.spectralEdges[0]:self.spectralEdges[1]
            ]
        # Step 1: Line Selections
        if self.lineSelectionMethod == "MANUAL":
            # Get an approximate reference spectrum from FTS
            apxWavemin = self.centralWavelength - np.nanmean(self.spectralEdges) * self.approxDispersion
            apxWavemax = self.centralWavelength + np.nanmean(self.spectralEdges) * self.approxDispersion
            # Fudge: Pad out FTS selection, since the central wavelength and dispersion are VERY approximate
            apxWavemin = apxWavemin - 50 * self.approxDispersion
            apxWavemax = apxWavemax + 50 * self.approxDispersion
            fts_wave, fts_spec = spex.fts_window(apxWavemin, apxWavemax)
            centralProfile = np.nanmedian(
                croppedFlat[
                    int(croppedFlat.shape[0]/2 - 30):int(croppedFlat.shape[0]/2 + 30), :
                ],
                axis=0
            )
            print("Top: HSG Spectrum (uncorrected). Bottom: FTS Reference Spectrum.")
            print("Select the same two spectral lines on each plot.")
            hsgLines, ftsLines = spex.select_lines_doublepanel(
                centralProfile,
                fts_spec,
                4
            )
            hsgLineCores = [
                int(spex.find_line_core(centralProfile[x - 5:x + 5]) + x + 5) for x in hsgLines
            ]
            ftslinecores = [
                spex.find_line_core(fts_spec[x - 5:x + 5]) + x + 5 for x in ftsLines
            ]
            # Step 1.5: Save reference profile and line centers for later
            self.lineSelectionIndices = hsgLineCores
            self.ftsSelectionIndices = ftslinecores
            self.referenceFTSwave = fts_wave
            self.referenceFTSspec = fts_spec

        # Step 2: Iterate gain table per selected line
        gains = []
        coarse_gains = []
        deskews = []
        gainmeans = []
        for line in self.lineSelectionIndices:
            g, cg, desk = spex.create_gaintables(
                croppedFlat,
                [line - 5, line + 5],
                hairline_positions=self.hairlines
            )
            gains.append(g)
            coarse_gains.append(cg)
            deskews.append(desk)
            gainmeans.append(np.nanmean(gain))
        # Step 3: Find the gain table with the mean nearest 1 (i.e., best table)
        gainidx = spex.find_nearest(np.array(gainmeans), 1)
        return


    def hsg_fringe_correction(self):
        """
        Sets up parameters used for Fourier fringe correction.
        Writing this one only after the rest of the pipeline has been tested.
        :return:
        """
        return

    def hsg_wavelength_calibration(self):
        """
        Performs wavelength calibration from gain-corrected, deskewed, average flat.
        :return:
        """
        croppedFlat = self.avgFlat - self.avgDark
        croppedFlat = croppedFlat[self.beamEdges[0]:self.beamEdges[1], self.spectralEdges[0]:self.spectralEdges[1]]
        croppedFlat = croppedFlat / self.solarGain
        deskewedFlat = np.zeros(croppedFlat.shape)
        for i in range(deskewedFlat.shape[0]):
            deskewedFlat[i, :] = scind.shift(
                croppedFlat[i, :],
                self.deskewShifts[i],
                mode='nearest'
            )
        self.deskewedFlat = deskewedFlat
        meanProfile = np.nanmean(
            deskewedFlat[int(deskewedFlat.shape[0]/2 - 30):int(deskewedFlat.shape[0]/2 + 30), :],
            axis=0
        )
        trueCenters = [
            spex.find_line_core(meanProfile[x - 7:x + 7]) + x + 7 for x in self.lineSelectionIndices
        ]
        ftsRefWvls = [
            scinterp.interp1d(
                np.arange(len(self.referenceFTSspec)),
                self.referenceFTSspec,
                kind='linear'
            )(x) for x in self.ftsSelectionIndices
        ]
        angstrom_per_pixel = np.abs(ftsRefWvls[1] - ftsRefWvls[0]) / np.abs(trueCenters[1] - trueCenters[0])
        zerowvl = ftsRefWvls[0] - (angstrom_per_pixel * trueCenters[0])
        self.wavelengthArray = (np.arange(0, len(meanProfile)) * angstrom_per_pixel) + zerowvl
        return


    def hsg_perform_scan_calibration(self, selectLine=True):
        """
        Perfoms calibration of science data, triggers the
        :param selectLine: bool
            If true, prompts the user to select lines for use in velocity products.
            Otherwise, only corrects products and saves with no additional data
        :return:
        """
        for i in tqdm.tqdm(range(len(self.scanList)), desc="Correcting Scans"):
            with fits.open(self.scanList[i]) as schdu:
                # Corrected data cubes are set up as ny, nx, nlambda
                # This is consistent with the FIRS L-1.5 data products.
                correctedDataCube = np.zeros(
                    (
                        self.beamEdges[1] - self.beamEdges[0],
                        len(schdu) - 1,
                        self.spectralEdges[1] - self.spectralEdges[0]
                    )
                )
                # Determine slit step size
                # This is not noted in the obs logs, but IS noted in the HSG header
                # Which stores the entire AIW script.
                # We'll do a quick parse of the script and grab the line with "step" in it.
                aiwScript = list(schdu[0].header['COMMENT'])
                # Lines with "#" are comment lines
                aiwSteps = [x for x in aiwScript if "step" in x and "#" not in x]
                # Relies on the somewhat naive assumption that there's only one step command in the script
                # And that the step size is the last thing in that line.
                # This should be okay -- sometimes the effort needed to generalize a thing is greater
                # than the effort needed to write a hack that will work 99.9999999% of the time.
                self.stepSpacing = float(aiwSteps[0].split(" ")[-1])
                for j in range(1, len(schdu)):
                    rasterStep = schdu[j].data
                    rasterStep = rasterStep - self.avgDark
                    if self.lampGain:
                        rasterStep = rasterStep / self.lampGain
                    rasterStep = rasterStep[
                        self.beamEdges[0]:self.beamEdges[1],
                         self.spectralEdges[0]:self.spectralEdges[1]
                    ]
                    rasterStep = rasterStep / self.solarGain
                    # Extra step here: applying the deskew from the gain table creation step.
                    # There should be no wavelength variation along the slit now.
                    deskewedRasterStep = np.zeros(rasterStep.shape)
                    for k in range(rasterStep.shape[0]):
                        correctedDataCube[k, j - 1, :] = scind.shift(
                            rasterStep[k, :],
                            self.deskewShifts[k],
                            mode='nearest'
                        )
                if selectLine:
                    vmaps = self.hsg_create_velocity_maps(correctedDataCube)
                    self.packageScan(correctedDataCube, vmapList=vmaps)
                else:
                    self.packageScan(correctedDataCube)
        return


    def hsg_create_velocity_maps(self, spectralCube):
        """
        Generates velocity maps of the spectral cube.
        If self.velocityMapLineIndex is None, it prompts the user to select lines from a
        gain-corrected flat field. These values are used as the rest frame velocity.
        It then sets the user-generated values to a class attribute to be used on the
        rest of the main calibration loop.
        :param spectralCube: numpy.ndarray
            Reduced datacube of the shape (ny, nx, nlambda)
        :return vmaps: list
            List of velocity maps in km/s for each line chosen.
        """
        if not self.velocityMapLineIndex:
            meanProfile = np.nanmean(
                self.deskewedFlat[int(self.deskewedFlat.shape[0]/2 - 30):int(self.deskewedFlat.shape[0]/2 + 30), :],
                axis=0
            )
            vmapCoarseIndices = spex.select_lines_singlepanel_unbound(meanProfile)
            vmapFineIndices = [
                spex.find_line_core(meanProfile[x - 7:x + 7]) + x + 7 for x  in vmapCoarseIndices
            ]
            self.velocityMapLineIndex = vmapFineIndices
        vmaps = []
        for index in self.velocityMapLineIndex:
            refwvl = scinterp.interp1d(
                np.arange(len(self.wavelengthArray)),
                self.wavelengthArray,
                kind='linear'
            )(index)
            vmap = np.zeros((spectralCube.shape[0], spectralCube.shape[1]))
            for i in range(spectralCube.shape[0]):
                for j in range(spectralCube.shape[1]):
                    cwvl = spex.find_line_core(
                        spectralCube[i, j, int(index - 10):int(index + 10)],
                        wvl=self.wavelengthArray[int(index - 10):int(index + 10)]
                    )
                    vmap[i, j] = c_kms * (cwvl - refwvl)/refwvl
            vmaps.append(vmap)
        return vmaps


    def packageScan(self, correctedScan, vmapList=None):
        """
        Packages corrected scans, along with the level-1.5 velocity products
        :param correctedScan:
        :param vmapList:
        :return:
        """
        return