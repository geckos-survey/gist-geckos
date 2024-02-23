import logging
import os
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from multiprocess import Process, Queue
from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary
from gistPipeline.emissionLines.pyGandalf import gandalf_util as gandalf
from gistPipeline.prepareTemplates import _prepareTemplates

# PHYSICAL CONSTANTS
C = np.float64(299792.458)  # km/s


"""
PURPOSE:
  This module executes the emission-line analysis of the pipeline. Basically, it acts as an
  interface between pipeline and the pyGandALF routine
  (ui.adsabs.harvard.edu/?#abs/2006MNRAS.366.1151S; ui.adsabs.harvard.edu/abs/2006MNRAS.369..529F;
  ui.adsabs.harvard.edu/abs/2019arXiv190604746B) by implementing the input/output as well as
  preparation of data and eventually calling pyGandALF.
"""


def workerGANDALF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for (
        spectra,
        error,
        stellar_kin,
        templates,
        logLam_galaxy,
        logLam_template,
        emi_file,
        redshift,
        velscale,
        int_disp,
        reddening,
        mdeg,
        for_errors,
        offset,
        velscale_ratio,
        i,
        nbins,
        npix,
        config,
        maskedSpaxel,
    ) in iter(inQueue.get, "STOP"):
        weights, emission_templates, bestfit, sol, esol = run_gandalf(
            spectra,
            error,
            stellar_kin,
            templates,
            logLam_galaxy,
            logLam_template,
            emi_file,
            redshift,
            velscale,
            int_disp,
            reddening,
            mdeg,
            for_errors,
            offset,
            velscale_ratio,
            i,
            nbins,
            npix,
            config,
            maskedSpaxel,
        )

        outQueue.put((i, weights, emission_templates, bestfit, sol, esol))


def run_gandalf(
    spectrum,
    error,
    stellar_kin,
    templates,
    logLam_galaxy,
    logLam_template,
    emi_file,
    redshift,
    velscale,
    int_disp,
    reddening,
    mdeg,
    for_errors,
    offset,
    velscale_ratio,
    i,
    nbins,
    npix,
    config,
    maskedSpaxel,
):
    """
    Calls the pyGandALF routine (ui.adsabs.harvard.edu/?#abs/2006MNRAS.366.1151S;
    ui.adsabs.harvard.edu/abs/2006MNRAS.369..529F; ui.adsabs.harvard.edu/abs/2019arXiv190604746B)
    """
    printStatus.progressBar(i, nbins, barLength=50)

    # Leave this hardcoded here!
    plot = False
    degree = -1
    quiet = True
    log10 = False

    if maskedSpaxel == False:
        try:
            # Get goodpixels and emission_setup
            goodpixels, emission_setup = getGoodpixelsEmissionSetup(
                config, redshift, velscale, logLam_galaxy, logLam_template, npix
            )

            # Initial guess on velocity: Use value relative to stellar kinematics
            for itm in np.arange(len(emission_setup)):
                emission_setup[itm].v = emission_setup[itm].v + stellar_kin[0]

            # Run GANDALF
            weights, emission_templates, bestfit, sol, esol = gandalf.gandalf(
                templates,
                spectrum,
                error,
                velscale,
                stellar_kin,
                emission_setup,
                logLam_galaxy[0],
                logLam_galaxy[1] - logLam_galaxy[0],
                goodpixels,
                degree,
                mdeg,
                int_disp,
                plot,
                quiet,
                log10,
                reddening,
                logLam_template[0],
                for_errors,
                velscale_ratio,
                offset,
            )

            return [weights, emission_templates, bestfit, sol, esol]

        except:
            return [-1, -1, -1, -1, -1]
    else:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]


def save_gandalf(
    config,
    emission_setup,
    idx_l,
    sol,
    esol,
    nlines,
    sol_gas_AoN,
    stellar_kin,
    offset,
    optimal_template,
    logLam_template,
    logLam_galaxy,
    goodpixels,
    cleaned_spectrum,
    nweights,
    emission_weights,
    for_errors,
    npix,
    n_templates,
    bestfit,
    emissionSpectra,
    reddening,
    currentLevel,
):
    """Saves all results to disk."""

    # ========================
    # SAVE RESULTS
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_gas_"
        + currentLevel
        + ".fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_gas_" + currentLevel + ".fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Data and error HDUs
    names = []
    lambdas = []
    for ii in idx_l:
        names.append(str(emission_setup[ii].name))
        lambdas.append(str(emission_setup[ii]._lambda))

    gandalfOutput = Table()
    gandalfErrorOutput = Table()
    for i in range(len(idx_l)):
        if np.any(sol_gas_AoN[:, i]) != 0.0:
            gandalfOutput[names[i] + "_" + lambdas[i] + "_F"] = sol[:, i * 4 + 0]
            gandalfOutput[names[i] + "_" + lambdas[i] + "_A"] = sol[:, i * 4 + 1]
            gandalfOutput[names[i] + "_" + lambdas[i] + "_V"] = sol[:, i * 4 + 2]
            gandalfOutput[names[i] + "_" + lambdas[i] + "_S"] = sol[:, i * 4 + 3]
            gandalfOutput[names[i] + "_" + lambdas[i] + "_AON"] = sol_gas_AoN[:, i]
            if for_errors:
                gandalfErrorOutput[names[i] + "_" + lambdas[i] + "_FERR"] = esol[
                    :, i * 4 + 0
                ]
                gandalfErrorOutput[names[i] + "_" + lambdas[i] + "_AERR"] = esol[
                    :, i * 4 + 1
                ]
                gandalfErrorOutput[names[i] + "_" + lambdas[i] + "_VERR"] = esol[
                    :, i * 4 + 2
                ]
                gandalfErrorOutput[names[i] + "_" + lambdas[i] + "_SERR"] = esol[
                    :, i * 4 + 3
                ]

    if reddening != None:
        gandalfOutput["EBmV_0"] = sol[:, len(idx_l) * 4]
        gandalfOutput["EBmV_1"] = sol[:, len(idx_l) * 4 + 1]
        if for_errors:
            gandalfErrorOutput["EBmVERR_0"] = esol[:, len(idx_l) * 4]
            gandalfErrorOutput["EBmVERR_1"] = esol[:, len(idx_l) * 4 + 1]

    dataHDU = fits.BinTableHDU(gandalfOutput)
    if for_errors:
        errorHDU = fits.BinTableHDU(gandalfErrorOutput)

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
    if for_errors:
        errorHDU = _auxiliary.saveConfigToHeader(errorHDU, config["GAS"])

    if for_errors:
        HDUList = fits.HDUList([priHDU, dataHDU, errorHDU])
    else:
        HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_gas_" + currentLevel + ".fits"
    )
    logging.info("Wrote: " + outfits)

    # ========================
    # SAVE BESTFIT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_gas-bestfit_"
        + currentLevel
        + ".fits"
    )
    printStatus.running(
        "Writing: "
        + config["GENERAL"]["RUN_ID"]
        + "_gas-bestfit_"
        + currentLevel
        + ".fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(name="BESTFIT", format=str(bestfit.shape[1]) + "D", array=bestfit)
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Extension 2: Table HDU with logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam_galaxy))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Extension 3: Table HDU with goodpix
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodpixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["GAS"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["GAS"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: "
        + config["GENERAL"]["RUN_ID"]
        + "_gas-bestfit_"
        + currentLevel
        + ".fits"
    )
    logging.info("Wrote: " + outfits)

    # ========================
    # SAVE EMISSION
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_gas-emission_"
        + currentLevel
        + ".fits"
    )
    printStatus.running(
        "Writing: "
        + config["GENERAL"]["RUN_ID"]
        + "_gas-emission_"
        + currentLevel
        + ".fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="EMISSION",
            format=str(emissionSpectra.shape[1]) + "D",
            array=emissionSpectra,
        )
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "EMISSION"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: "
        + config["GENERAL"]["RUN_ID"]
        + "_gas-emission_"
        + currentLevel
        + ".fits"
    )
    logging.info("Wrote: " + outfits)

    # ========================
    # SAVE CLEANED SPECTRA
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_gas-cleaned_"
        + currentLevel
        + ".fits"
    )
    printStatus.running(
        "Writing: "
        + config["GENERAL"]["RUN_ID"]
        + "_gas-cleaned_"
        + currentLevel
        + ".fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(name="SPEC", format=str(npix) + "D", array=cleaned_spectrum)
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "CLEANED_SPECTRA"

    # Extension 2: Table HDU with logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam_galaxy))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["GAS"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: "
        + config["GENERAL"]["RUN_ID"]
        + "_gas-cleaned_"
        + currentLevel
        + ".fits"
    )
    logging.info("Wrote: " + outfits)

    # ========================
    # SAVE WEIGHTS
    if (config["GAS"]["LEVEL"] in ["BIN", "SPAXEL"]) or (
        config["GAS"]["LEVEL"] == "BOTH" and currentLevel == "BIN"
    ):
        outfits = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas-weights_"
            + currentLevel
            + ".fits"
        )
        printStatus.running(
            "Writing: "
            + config["GENERAL"]["RUN_ID"]
            + "_gas-weights_"
            + currentLevel
            + ".fits"
        )

        # Primary HDU
        priHDU = fits.PrimaryHDU()

        # Extension 1: Table HDU with normalized weights
        cols = []
        cols.append(
            fits.Column(name="NWEIGHTS", format=str(n_templates) + "D", array=nweights)
        )
        dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        dataHDU.name = "NWEIGHTS"

        # Extension 2: Table HDU with weights of emission-lines
        cols = []
        cols.append(
            fits.Column(
                name="EWEIGHTS",
                format=str(emission_weights.shape[1]) + "D",
                array=emission_weights,
            )
        )
        eweightsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        eweightsHDU.name = "EWEIGHTS"

        # Create HDU list and write to file
        priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
        dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
        eweightsHDU = _auxiliary.saveConfigToHeader(eweightsHDU, config["GAS"])

        HDUList = fits.HDUList([priHDU, dataHDU, eweightsHDU])
        HDUList.writeto(outfits, overwrite=True)

        printStatus.updateDone(
            "Writing: "
            + config["GENERAL"]["RUN_ID"]
            + "_gas-weights_"
            + currentLevel
            + ".fits"
        )
        logging.info("Wrote: " + outfits)

    # ========================
    # SAVE OPTIMAL TEMPLATE
    if currentLevel == "BIN":
        outfits = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas-optimalTemplate_"
            + currentLevel
            + ".fits"
        )
        printStatus.running(
            "Writing: "
            + config["GENERAL"]["RUN_ID"]
            + "_gas-optimalTemplate_"
            + currentLevel
            + ".fits"
        )

        # Primary HDU
        priHDU = fits.PrimaryHDU()

        # Extension 1: Table HDU with optimal templates
        cols = []
        cols.append(
            fits.Column(
                name="OPTIMAL_TEMPLATE",
                format=str(optimal_template.shape[1]) + "D",
                array=optimal_template,
            )
        )
        dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        dataHDU.name = "OPTIMAL_TEMPLATE"

        # Extension 2: Table HDU with logLam_templates
        cols = []
        cols.append(
            fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template)
        )
        logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        logLamHDU.name = "LOGLAM_TEMPLATE"

        # Create HDU list and write to file
        priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
        dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
        logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["GAS"])

        HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
        HDUList.writeto(outfits, overwrite=True)

        printStatus.updateDone(
            "Writing: "
            + config["GENERAL"]["RUN_ID"]
            + "_gas-optimalTemplate_"
            + currentLevel
            + ".fits"
        )
        logging.info("Wrote: " + outfits)


def getGoodpixelsEmissionSetup(
    config, redshift, velscale, logLam_galaxy, logLam_template, npix
):
    """
    Reads information about the emission-lines from the emissionLines.config
    file, in order to
     * determine which spectral regions are masked in the GandALF fit
     * create a structure which states details on the intended emission-line
       fits
    """
    # Read in emission-line setup file
    eml_file = os.path.join(config["GENERAL"]["CONFIG_DIR"], config["GAS"]["EMI_FILE"])
    eml_i = np.genfromtxt(eml_file, dtype="int", usecols=(0), comments="#")
    eml_name = np.genfromtxt(eml_file, dtype="str", usecols=(1), comments="#")
    eml_lambda = np.genfromtxt(eml_file, dtype="float", usecols=(2), comments="#")
    eml_action = np.genfromtxt(eml_file, dtype="str", usecols=(3), comments="#")
    eml_kind = np.genfromtxt(eml_file, dtype="str", usecols=(4), comments="#")
    eml_a = np.genfromtxt(eml_file, dtype="float", usecols=(5), comments="#")
    eml_v = np.genfromtxt(eml_file, dtype="int", usecols=(6), comments="#")
    eml_s = np.genfromtxt(eml_file, dtype="int", usecols=(7), comments="#")
    eml_fit = np.genfromtxt(eml_file, dtype="str", usecols=(8), comments="#")
    eml_aon = np.genfromtxt(eml_file, dtype="int", usecols=(9), comments="#")

    emission_setup = gandalf.load_emission_setup_new(
        np.vstack(
            (
                eml_i,
                eml_name,
                eml_lambda,
                eml_action,
                eml_kind,
                eml_a,
                eml_v,
                eml_s,
                eml_fit,
                eml_aon,
            )
        )
    )

    # Create goodpixels and final emission line setup
    goodpixels, emission_setup = gandalf.mask_emission_lines(
        npix,
        redshift,
        emission_setup,
        velscale,
        logLam_galaxy[0],
        (logLam_galaxy[1] - logLam_galaxy[0]),
        None,
        None,
        0,
        config["GENERAL"]["REDSHIFT"],
    )

    # Prepare emission_setup structure for GANDALF, which should only deal with the lines we fit
    i_f = []
    for itm in np.arange(len(emission_setup)):
        if emission_setup[itm].action != "f":
            i_f.append(itm)
    emission_setup = [
        v for i, v in enumerate(emission_setup) if i not in frozenset(i_f)
    ]

    return (goodpixels, emission_setup)


def performEmissionLineAnalysis(config):
    """
    Starts the emission-line analysis. Input data is read from file on either
    bin or spaxel level. The LSF is loaded and spectra can be de-reddened for
    Galactic extinction in the direction of the target. Spectral pixels and
    emission-lines considered in the fit are determined. After the pyGandALF
    fit, emission-subtracted spectral are calculated. Results are saved to disk.
    """

    #    # Check if the error estimation in pyGandalf is turned off
    #    if config['GAS']['ERRORS'] != 0:
    #        printStatus.warning("It is currently not possible to derive errors with pyGandALF in a Python3 environment. An updated version of pyGandALF will be released soon.")
    #        printStatus.warning("The emission-line analysis continues without an error estimation.")
    #        logging.warning("It is currently not possible to derive errors with pyGandALF in a Python3 environment. An updated version of pyGandALF will be released soon.")
    #        logging.warning("The emission-line analysis continues without an error estimation.")
    #        config['GAS']['ERRORS'] = 0

    # Check if proper configuration is set
    if config["GAS"]["LEVEL"] not in ["BIN", "SPAXEL", "BOTH"]:
        message = "Configuration parameter GAS|SPAXEL has to be either 'BIN', 'SPAXEL', or 'BOTH'."
        printStatus.failed(message)
        logging.error(message)
        raise Exception(message)

    # For output filenames
    if config["GAS"]["LEVEL"] in ["BIN", "SPAXEL"]:
        currentLevel = config["GAS"]["LEVEL"]
    elif (
        config["GAS"]["LEVEL"] == "BOTH"
        and os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas_BIN.fits"
        )
        == False
    ):
        currentLevel = "BIN"
    elif (
        config["GAS"]["LEVEL"] == "BOTH"
        and os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas_BIN.fits"
        )
        == True
    ):
        currentLevel = "SPAXEL"

    # Oversample the templates by a factor of two
    velscale_ratio = 2

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "GAS")

    # Read data if we run on BIN level
    if currentLevel == "BIN":
        # Read spectra from file
        hdu = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_BinSpectra.fits"
        )
        spectra = np.array(hdu[1].data.SPEC.T)
        logLam_galaxy = np.array(hdu[2].data.LOGLAM)
        idx_lam = np.where(
            np.logical_and(
                np.exp(logLam_galaxy) > config["GAS"]["LMIN"],
                np.exp(logLam_galaxy) < config["GAS"]["LMAX"],
            )
        )[0]
        spectra = spectra[idx_lam, :]
        logLam_galaxy = logLam_galaxy[idx_lam]
        npix = spectra.shape[0]
        nbins = spectra.shape[1]
        velscale = hdu[0].header["VELSCALE"]

        # Create empty mask in bin-level run: There are no masked bins, only masked spaxels!
        maskedSpaxel = np.zeros(nbins, dtype=bool)
        maskedSpaxel[:] = False

        # Prepare templates
        logging.info("Using full spectral library for GANDALF on BIN level")
        (
            templates,
            lamRange_spmod,
            logLam_template,
            n_templates,
        ) = _prepareTemplates.prepareTemplates_Module(
            config,
            config["GAS"]["LMIN"],
            config["GAS"]["LMAX"],
            velscale / velscale_ratio,
            LSF_Data,
            LSF_Templates,
            "GAS",
        )[
            :4
        ]
        templates = templates.reshape((templates.shape[0], n_templates))

        offset = (logLam_template[0] - logLam_galaxy[0]) * C  # km/s
        error = np.ones((npix, nbins))

        # Read stellar kinematics from file
        ppxf = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits"
        )[1].data
        stellar_kin = np.zeros((ppxf.V.shape[0], 6))
        stellar_kin[:, 0] = np.array(ppxf.V)
        stellar_kin[:, 1] = np.array(ppxf.SIGMA)
        stellar_kin[:, 2] = np.array(ppxf.H3)
        stellar_kin[:, 3] = np.array(ppxf.H4)

        # Rename to keep the code clean
        for_errors = config["GAS"]["ERRORS"]

    # Read data if we run on SPAXEL level
    elif currentLevel == "SPAXEL":
        # Read spectra from file
        hdu = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_AllSpectra.fits"
        )
        spectra = np.array(hdu[1].data.SPEC.T)
        logLam_galaxy = np.array(hdu[2].data.LOGLAM)
        idx_lam = np.where(
            np.logical_and(
                np.exp(logLam_galaxy) > config["GAS"]["LMIN"],
                np.exp(logLam_galaxy) < config["GAS"]["LMAX"],
            )
        )[0]
        spectra = spectra[idx_lam, :]
        logLam_galaxy = logLam_galaxy[idx_lam]
        npix = spectra.shape[0]
        nbins = spectra.shape[1]
        velscale = hdu[0].header["VELSCALE"]

        # Construct mask for defunct spaxels
        mask = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_mask.fits"
        )[1].data.MASK_DEFUNCT
        maskedSpaxel = np.array(mask, dtype=bool)

        # Prepare templates
        if config["GAS"]["LEVEL"] == "SPAXEL":
            logging.info("Using full spectral library for GANDALF on SPAXEL level")
            (
                templates,
                lamRange_spmod,
                logLam_template,
                n_templates,
            ) = _prepareTemplates.prepareTemplates_Module(
                config,
                config["GAS"]["LMIN"],
                config["GAS"]["LMAX"],
                velscale / velscale_ratio,
                LSF_Data,
                LSF_Templates,
                "GAS",
            )[
                :4
            ]
            templates = templates.reshape((templates.shape[0], n_templates))
        if config["GAS"]["LEVEL"] == "BOTH":
            logging.info(
                "Using previously extracted optimal templates from the GANDALF BIN level on SPAXEL level"
            )
            hdu = fits.open(
                os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                + "_gas-optimalTemplate_BIN.fits"
            )
            templates = np.array(hdu[1].data.OPTIMAL_TEMPLATE.T)
            logLam_template = np.array(hdu[2].data.LOGLAM_TEMPLATE)
            n_templates = 1
            printStatus.done("Preparing the stellar population templates")
        offset = (logLam_template[0] - logLam_galaxy[0]) * C  # km/s
        error = np.ones((npix, nbins))

        # Read stellar kinematics from file
        ppxf = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits"
        )[1].data
        stellar_kin = np.zeros((ppxf.V.shape[0], 6))
        stellar_kin[:, 0] = np.array(ppxf.V)
        stellar_kin[:, 1] = np.array(ppxf.SIGMA)
        stellar_kin[:, 2] = np.array(ppxf.H3)
        stellar_kin[:, 3] = np.array(ppxf.H4)

        # Read bintable
        bintable = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_table.fits"
        )[1].data
        binNum_long = np.array(bintable.BIN_ID)
        ubins = np.unique(np.abs(binNum_long))

        # Convert stellar kinematics to long version
        stellar_kin_long = np.zeros((len(binNum_long), stellar_kin.shape[1]))
        if n_templates == 1:
            templates_long = np.zeros((len(binNum_long), templates.shape[0]))
        for i in range(0, len(ubins)):
            idx = np.where(ubins[i] == np.abs(binNum_long))[0]
            stellar_kin_long[idx, :] = stellar_kin[i, :]
            if n_templates == 1:
                templates_long[idx, :] = templates[:, i]
        stellar_kin = stellar_kin_long
        if n_templates == 1:
            templates = templates_long

        # Rename to keep the code clean
        for_errors = config["GAS"]["ERRORS"]

    # LSF function for emission templates
    int_disp = np.zeros((2, npix))
    int_disp[0, :] = np.exp(logLam_galaxy)
    int_disp[1, :] = C * (
        np.array(LSF_Data(np.exp(logLam_galaxy))) / 2.355 / np.exp(logLam_galaxy)
    )

    # Deredden the spectra for the Galactic extinction in the direction of the target
    if config["GAS"]["EBmV"] != None:
        dereddening_attenuation = gandalf.dust_calzetti(
            logLam_galaxy[0],
            logLam_galaxy[1] - logLam_galaxy[0],
            npix,
            -1 * config["GAS"]["EBmV"],
            0.0,
            0,
        )
        for i in range(spectra.shape[1]):
            spectra[:, i] = spectra[:, i] * dereddening_attenuation

    # Get goodpixels and emission_setup
    goodpixels, emission_setup = getGoodpixelsEmissionSetup(
        config, 0.0, velscale, logLam_galaxy, logLam_template, npix
    )

    # Setup output arrays
    nlines = 0
    for itm in np.arange(len(emission_setup)):
        if emission_setup[itm].action == "f" and emission_setup[itm].kind == "l":
            nlines += 1

    if config["GAS"]["REDDENING"] == False:
        reddening = None
        mdegree = config["GAS"]["MDEG"]
        reddening_length_sol = config["GAS"]["MDEG"]
        reddening_length_esol = 0
    else:
        reddening = [
            float(config["GAS"]["REDDENING"].split(",")[0].strip()),
            float(config["GAS"]["REDDENING"].split(",")[1].strip()),
        ]
        mdegree = 0
        reddening_length_sol = 2
        reddening_length_esol = 2

    weights = np.zeros((nbins, n_templates + nlines))
    emission_templates = np.zeros((nbins, nlines, npix))
    bestfit = np.zeros((nbins, npix))
    sol = np.zeros((nbins, nlines * 4 + reddening_length_sol))
    esol = np.zeros((nbins, nlines * 4 + reddening_length_esol))

    nweights = np.zeros((nbins, n_templates))
    optimal_template = np.zeros((nbins, templates.shape[0]))
    sol_gas_AoN = np.zeros((nbins, nlines))
    cleaned_spectrum = np.zeros((nbins, npix))

    # ========================
    # Run GANDALF
    start_time = time.time()

    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running GANDALF in parallel mode")
        logging.info("Running GANDALF in parallel mode")

        # Create Queues
        inQueue = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [
            Process(target=workerGANDALF, args=(inQueue, outQueue))
            for _ in range(config["GENERAL"]["NCPU"])
        ]

        # Start worker processes
        for p in ps:
            p.start()

        # Fill the queue
        if n_templates > 1:
            for i in range(nbins):
                inQueue.put(
                    (
                        spectra[:, i],
                        error[:, i],
                        stellar_kin[i, :],
                        templates,
                        logLam_galaxy,
                        logLam_template,
                        config["GAS"]["EMI_FILE"],
                        0.0,
                        velscale,
                        int_disp,
                        reddening,
                        mdegree,
                        for_errors,
                        offset,
                        velscale_ratio,
                        i,
                        nbins,
                        npix,
                        config,
                        maskedSpaxel[i],
                    )
                )
        elif n_templates == 1:
            for i in range(nbins):
                inQueue.put(
                    (
                        spectra[:, i],
                        error[:, i],
                        stellar_kin[i, :],
                        templates[[i], :].T,
                        logLam_galaxy,
                        logLam_template,
                        config["GAS"]["EMI_FILE"],
                        0.0,
                        velscale,
                        int_disp,
                        reddening,
                        mdegree,
                        for_errors,
                        offset,
                        velscale_ratio,
                        i,
                        nbins,
                        npix,
                        config,
                        maskedSpaxel[i],
                    )
                )

        # now get the results with indices
        gandalf_tmp = [outQueue.get() for _ in range(nbins)]

        # send stop signal to stop iteration
        for _ in range(config["GENERAL"]["NCPU"]):
            inQueue.put("STOP")

        # stop processes
        for p in ps:
            p.join()

        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i] = gandalf_tmp[i][0]
            weights[i, :] = gandalf_tmp[i][1]
            emission_templates[i, :, :] = gandalf_tmp[i][2]
            bestfit[i, :] = gandalf_tmp[i][3]
            sol[i, :] = gandalf_tmp[i][4]
            esol[i, :] = gandalf_tmp[i][5]
        # Sort output
        argidx = np.argsort(index)
        weights = weights[argidx, :]
        emission_templates = emission_templates[argidx, :, :]
        bestfit = bestfit[argidx, :]
        sol = sol[argidx, :]
        esol = esol[argidx, :]

        printStatus.updateDone("Running GANDALF in parallel mode", progressbar=True)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running GANDALF in serial mode")
        logging.info("Running GANDALF in serial mode")
        if n_templates > 1:
            for i in range(0, nbins):
                (
                    weights[i, :],
                    emission_templates[i, :, :],
                    bestfit[i, :],
                    sol[i, :],
                    esol[i, :],
                ) = run_gandalf(
                    spectra[:, i],
                    error[:, i],
                    stellar_kin[i, :],
                    templates,
                    logLam_galaxy,
                    logLam_template,
                    config["GAS"]["EMI_FILE"],
                    0.0,
                    velscale,
                    int_disp,
                    reddening,
                    mdegree,
                    for_errors,
                    offset,
                    velscale_ratio,
                    i,
                    nbins,
                    npix,
                    config,
                    maskedSpaxel[i],
                )
        elif n_templates == 1:
            for i in range(0, nbins):
                (
                    weights[i, :],
                    emission_templates[i, :, :],
                    bestfit[i, :],
                    sol[i, :],
                    esol[i, :],
                ) = run_gandalf(
                    spectra[:, i],
                    error[:, i],
                    stellar_kin[i, :],
                    templates[[i], :].T,
                    logLam_galaxy,
                    logLam_template,
                    config["GAS"]["EMI_FILE"],
                    0.0,
                    velscale,
                    int_disp,
                    reddening,
                    mdegree,
                    for_errors,
                    offset,
                    velscale_ratio,
                    i,
                    nbins,
                    npix,
                    config,
                    maskedSpaxel[i],
                )

        printStatus.updateDone("Running GANDALF in serial mode", progressbar=True)

    print(
        "             Running GANDALF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )
    logging.info(
        "Running GANDALF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check for exceptions which occurred during the analysis
    idx_error = np.where(bestfit[:, 0] == -1)[0]
    if len(idx_error) != 0:
        printStatus.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
        )
        print("             " + str(idx_error))
        logging.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
            + str(idx_error)
        )
    else:
        print("             " + "There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Get AoN thresholds from emission_setup
    AoN_thresholds = np.zeros(nlines)
    o = 0
    for itm in np.arange(len(emission_setup)):
        if emission_setup[itm].action == "f" and emission_setup[itm].kind == "l":
            AoN_thresholds[o] = emission_setup[itm].aon
            o = o + 1

    # Get optimal template and calculate emission-subtracted spectra
    # TODO: This should be done within the parallel loop
    idx_l = gandalf.where_eq(emission_setup, "kind", "l")
    emission_templates = np.transpose(emission_templates, (0, 2, 1))
    for i in range(0, nbins):
        if (config["GAS"]["LEVEL"] in ["BIN", "SPAXEL"]) or (
            config["GAS"]["LEVEL"] == "BOTH" and currentLevel == "BIN"
        ):
            # Make the unconvolved optimal stellar template
            nweights[i, :] = weights[i, :n_templates] / np.sum(weights[i, :n_templates])
            for j in range(0, n_templates):
                optimal_template[i, :] = (
                    optimal_template[i, :] + templates[:, j] * nweights[i, j]
                )

        # Calculate emission-subtracted spectra using a AoN threshold
        sol_gas_A = sol[
            i, np.arange(len(idx_l)) * 4 + 1
        ]  # Array of solutions for amplitudes
        sol_gas_AoN[i, :], cleaned_spectrum[i, :] = gandalf.remouve_detected_emission(
            spectra[:, i],
            bestfit[i, :],
            emission_templates[i, :, :],
            sol_gas_A,
            AoN_thresholds,
            None,
        )

    # Calculate the best fitting emission
    emissionSpectrum = np.sum(emission_templates, axis=2)
    emissionSubtractedBestfit = bestfit - emissionSpectrum

    # Save results to file
    save_gandalf(
        config,
        emission_setup,
        idx_l,
        sol,
        esol,
        nlines,
        sol_gas_AoN,
        stellar_kin,
        offset,
        optimal_template,
        logLam_template,
        logLam_galaxy,
        goodpixels,
        cleaned_spectrum,
        nweights,
        weights[:, n_templates:],
        for_errors,
        npix,
        n_templates,
        bestfit,
        emissionSpectrum,
        reddening,
        currentLevel,
    )

    # Restart GANDALF if a SPAXEL level run based on a previous BIN level run is intended
    if config["GAS"]["LEVEL"] == "BOTH" and currentLevel == "BIN":
        print()
        performEmissionLineAnalysis(config)

    # Return
    return None
