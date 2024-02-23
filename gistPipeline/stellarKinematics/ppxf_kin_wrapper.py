import code
import logging
import os
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from joblib import Parallel, delayed, dump, load
from ppxf.ppxf import ppxf
from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary
from gistPipeline.prepareTemplates import _prepareTemplates

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE:
  This module executes the analysis of stellar kinematics in the pipeline.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""


def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    y = np.ravel(y)
    d = y if zero else y - np.median(y)

    mad = np.median(np.abs(d))
    u2 = (d / (9.0 * mad)) ** 2  # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good] * u1**2) ** 2).sum()
    den = (u1 * (1.0 - 5.0 * u2[good])).sum()
    sigma = np.sqrt(num / (den * (den - 1.0)))  # see note in above reference

    return sigma

def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    adeg,
    mdeg,
    reddening,
    doclean,
    logLam,
    offset,
    velscale_ratio,
    nsims,
    nbins,
    i,
    optimal_template_in,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics.
    """
    printStatus.progressBar(i, nbins, barLength=50)

    try:
        # Call PPXF for first time to get optimal template
        if len(optimal_template_in) == 1:
            print("Running pPXF for the first time")
            pp = ppxf(
                templates,
                log_bin_data,
                log_bin_error,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=True,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )
        else:
            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise = np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

            # Find a proper estimate of the noise
            noise_orig = biweight_location(log_bin_error[goodPixels])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels] - pp_step1.bestfit[goodPixels]
            )

            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error * (noise_est / noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 2 ##################
            # Second Call PPXF - use best-fitting template, determine outliers
            # only do this if doclean is set
            if doclean == True:
                pp_step2 = ppxf(
                    optimal_template_in,
                    log_bin_data,
                    noise_new,
                    velscale,
                    start,
                    goodpixels=goodPixels,
                    plot=False,
                    quiet=True,
                    moments=nmoments,
                    degree=adeg,
                    mdegree=mdeg,
                    reddening=reddening,
                    lam=np.exp(logLam),
                    velscale_ratio=velscale_ratio,
                    vsyst=offset,
                    clean=True,
                )

                # update goodpixels
                goodPixels = pp_step2.goodpixels

                # repeat noise scaling # Find a proper estimate of the noise
                noise_orig = biweight_location(log_bin_error[goodPixels])
                noise_est = robust_sigma(
                    pp_step2.galaxy[goodPixels] - pp_step2.bestfit[goodPixels]
                )

                # Calculate the new noise, and the sigma of the distribution.
                noise_new = log_bin_error * (noise_est / noise_orig)
                noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel
            # causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 3 ##################
            # Third Call PPXF - use all templates, get best-fit
            pp = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

        # update goodpixels again
        goodPixels = pp.goodpixels

        # make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # Calculate the true S/N from the residual
        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmean(pp.galaxy[goodPixels]/noise_est)

        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum(pp.weights)
        optimal_template = np.zeros(templates.shape[0])
        for j in range(0, templates.shape[1]):
            optimal_template = (
                optimal_template + templates[:, j] * normalized_weights[j]
            )

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        # Do MC-Simulations
        sol_MC = np.zeros((nsims, nmoments))
        mc_results = np.zeros(nmoments)
        for o in range(0, nsims):
            # Add noise to bestfit:
            #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
            #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
            noisy_bestfit = pp.bestfit + np.random.normal(
                0, 1, len(log_bin_data)
            ) * np.std(log_bin_data[goodPixels] - pp.bestfit[goodPixels])

            mc = ppxf(
                templates,
                noisy_bestfit,
                log_bin_error,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                velscale_ratio=velscale_ratio,
                vsyst=offset,
                bias=0.0,
            )
            sol_MC[o, :] = mc.sol[:]

        if nsims != 0:
            mc_results = np.nanstd(sol_MC, axis=0)

        return(
            pp.sol[:],
            pp.reddening,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
        )

    except:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def save_ppxf(
    config,
    ppxf_result,
    ppxf_reddening,
    mc_results,
    formal_error,
    ppxf_bestfit,
    logLam,
    goodPixels,
    optimal_template,
    logLam_template,
    npix,
    spectral_mask,
    optimal_template_comb,
    bin_data,
    snr_postfit,
):
    """Saves all results to disk."""
    # ========================
    # SAVE RESULTS
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF output data
    cols = []
    cols.append(fits.Column(name="V", format="D", array=ppxf_result[:, 0]))
    cols.append(fits.Column(name="SIGMA", format="D", array=ppxf_result[:, 1]))
    if np.any(ppxf_result[:, 2]) != 0:
        cols.append(fits.Column(name="H3", format="D", array=ppxf_result[:, 2]))
    if np.any(ppxf_result[:, 3]) != 0:
        cols.append(fits.Column(name="H4", format="D", array=ppxf_result[:, 3]))
    if np.any(ppxf_result[:, 4]) != 0:
        cols.append(fits.Column(name="H5", format="D", array=ppxf_result[:, 4]))
    if np.any(ppxf_result[:, 5]) != 0:
        cols.append(fits.Column(name="H6", format="D", array=ppxf_result[:, 5]))

    if np.any(mc_results[:, 0]) != 0:
        cols.append(fits.Column(name="ERR_V", format="D", array=mc_results[:, 0]))
    if np.any(mc_results[:, 1]) != 0:
        cols.append(fits.Column(name="ERR_SIGMA", format="D", array=mc_results[:, 1]))
    if np.any(mc_results[:, 2]) != 0:
        cols.append(fits.Column(name="ERR_H3", format="D", array=mc_results[:, 2]))
    if np.any(mc_results[:, 3]) != 0:
        cols.append(fits.Column(name="ERR_H4", format="D", array=mc_results[:, 3]))
    if np.any(mc_results[:, 4]) != 0:
        cols.append(fits.Column(name="ERR_H5", format="D", array=mc_results[:, 4]))
    if np.any(mc_results[:, 5]) != 0:
        cols.append(fits.Column(name="ERR_H6", format="D", array=mc_results[:, 5]))

    cols.append(fits.Column(name="FORM_ERR_V", format="D", array=formal_error[:, 0]))
    cols.append(
        fits.Column(name="FORM_ERR_SIGMA", format="D", array=formal_error[:, 1])
    )
    if np.any(formal_error[:, 2]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H3", format="D", array=formal_error[:, 2])
        )
    if np.any(formal_error[:, 3]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H4", format="D", array=formal_error[:, 3])
        )
    if np.any(formal_error[:, 4]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H5", format="D", array=formal_error[:, 4])
        )
    if np.any(formal_error[:, 5]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H6", format="D", array=formal_error[:, 5])
        )

    # Add reddening if parameter is used
    if np.any(np.isnan(ppxf_reddening)) != True:
        cols.append(fits.Column(name="REDDENING", format="D", array=ppxf_reddening[:]))

    # Add True SNR calculated from residual
    cols.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "KIN_DATA"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin.fits")
    logging.info("Wrote: " + outfits_ppxf)

    # ========================
    # SAVE BESTFIT
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name="BESTFIT", format=str(npix) + "D", array=ppxf_bestfit))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Table HDU with PPXF logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Table HDU with ??? --> unclear what this is?
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data.T))
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = "SPEC"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["KIN"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["KIN"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["KIN"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU, specHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit.fits"
    )
    logging.info("Wrote: " + outfits_ppxf)

    # ============================
    # SAVE OPTIMAL TEMPLATE RESULT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-optimalTemplates.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-optimalTemplates.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATES",
            format=str(optimal_template.shape[1]) + "D",
            array=optimal_template,
        )
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "OPTIMAL_TEMPLATES"

    # Extension 2: Table HDU with logLam_templates
    cols = []
    cols.append(fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM_TEMPLATE"

    # Extension 2: Table HDU with logLam_templates
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATE_ALL", format="D", array=optimal_template_comb
        )
    )
    combHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    combHDU.name = "OPTIMAL_TEMPLATE_ALL"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["KIN"])
    combHDU = _auxiliary.saveConfigToHeader(combHDU, config["KIN"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, combHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-optimalTemplates.fits"
    )
    logging.info("Wrote: " + outfits)

    # ============================
    # SAVE SPECTRAL MASK RESULT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-SpectralMask.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-SpectralMask.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="SPECTRAL_MASK",
            format=str(spectral_mask.shape[1]) + "D",
            array=spectral_mask,
        )
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "SPECTRAL_MASK"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-SpectralMask.fits"
    )
    logging.info("Wrote: " + outfits)

def extractStellarKinematics(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and
    saves the outputs following the GIST conventions.
    """
    # Read data from file
    infile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5"
    printStatus.running("Reading: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.hdf5")
    
    # Open the HDF5 file
    with h5py.File(infile, 'r') as f:
        
        # Read the data from the file
        logLam = f['LOGLAM'][:]
        idx_lam = np.where(
        np.logical_and(
            np.exp(logLam) > config["KIN"]["LMIN"],
            np.exp(logLam) < config["KIN"]["LMAX"],
        )
        )[0]

        bin_data = f['SPEC'][:][idx_lam, :]
        bin_err = f['ESPEC'][:][idx_lam, :]
        velscale = f.attrs['VELSCALE']
    
    logLam = logLam[idx_lam]
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)

    # Read LSF information

    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "KIN")  # added input of module

    # Prepare templates
    velscale_ratio = 2
    logging.info("Using full spectral library for PPXF")
    (
        templates,
        lamRange_spmod,
        logLam_template,
        ntemplates,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        config["KIN"]["LMIN"],
        config["KIN"]["LMAX"],
        velscale / velscale_ratio,
        LSF_Data,
        LSF_Templates,
        "KIN",
    )[
        :4
    ]
    templates = templates.reshape((templates.shape[0], ntemplates))

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0]) * C
    # noise  = np.ones((npix,nbins))
    noise = bin_err  # is actual noise, not variance
    nsims = config["KIN"]["MC_PPXF"]

    # Initial guesses
    start = np.zeros((nbins, 2))
    if (
        os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits"
        )
        == True
    ):
        printStatus.done(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin-guess.fits' as initial guesses"
        )
        logging.info(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin-guess.fits' as initial guesses"
        )
        guess = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits"
        )[1].data
        start[:, 0] = guess.V
        start[:, 1] = guess.SIGMA
    else:
        # Use the same initial guess for all bins, as stated in MasterConfig
        printStatus.done(
            "Using V and SIGMA from the MasterConfig file as initial guesses"
        )
        logging.info("Using V and SIGMA from the MasterConfig file as initial guesses")
        start[:, 0] = 0.0
        start[:, 1] = config["KIN"]["SIGMA"]

    # Define goodpixels
    goodPixels_ppxf = _auxiliary.spectralMasking(
        config, config["KIN"]["SPEC_MASK"], logLam
    )

    # Array to store results of ppxf
    ppxf_result = np.zeros((nbins, 6))
    ppxf_reddening = np.zeros(nbins)
    ppxf_bestfit = np.zeros((nbins, npix))
    optimal_template = np.zeros((nbins, templates.shape[0]))
    mc_results = np.zeros((nbins, 6))
    formal_error = np.zeros((nbins, 6))
    spectral_mask = np.zeros((nbins, bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)

    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:, :], axis=1)
    comb_espec = np.nanmean(bin_err[:, :], axis=1)
    optimal_template_init = [0]

    (
        tmp_ppxf_result,
        tmp_ppxf_reddening,
        tmp_ppxf_bestfit,
        optimal_template_out,
        tmp_mc_results,
        tmp_formal_error,
        tmp_spectral_mask,
        tmp_snr_postfit,
    ) = run_ppxf(
        templates,
        comb_spec,
        comb_espec,
        velscale,
        start[0, :],
        goodPixels_ppxf,
        config["KIN"]["MOM"],
        config["KIN"]["ADEG"],
        config["KIN"]["MDEG"],
        config["KIN"]["REDDENING"],
        config["KIN"]["DOCLEAN"],
        logLam,
        offset,
        velscale_ratio,
        nsims,
        nbins,
        0,
        optimal_template_init,
    )
    # now define the optimal template that we'll use throughout
    optimal_template_comb = optimal_template_out

    # ====================
    # Run PPXF
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")
        
        # Prepare the folder where the memmap will be dumped
        memmap_folder = "/scratch" if os.access("/scratch", os.W_OK) else config["GENERAL"]["OUTPUT"]

        # dump the arrays and load as memmap
        templates_filename_memmap = memmap_folder + "/templates_memmap.tmp"
        dump(templates, templates_filename_memmap)
        templates = load(templates_filename_memmap, mmap_mode='r')
        
        bin_data_filename_memmap = memmap_folder + "/bin_data_memmap.tmp"
        dump(bin_data, bin_data_filename_memmap)
        bin_data = load(bin_data_filename_memmap, mmap_mode='r')
        
        noise_filename_memmap = memmap_folder + "/noise_memmap.tmp"
        dump(noise, noise_filename_memmap)
        noise = load(noise_filename_memmap, mmap_mode='r')

        # Define a function to encapsulate the work done in the loop
        def worker(chunk, templates):
            results = []
            for i in chunk:
                result = run_ppxf(
                    templates,
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    goodPixels_ppxf,
                    config["KIN"]["MOM"],
                    config["KIN"]["ADEG"],
                    config["KIN"]["MDEG"],
                    config["KIN"]["REDDENING"],
                    config["KIN"]["DOCLEAN"],
                    logLam,
                    offset,
                    velscale_ratio,
                    nsims,
                    nbins,
                    i,
                    optimal_template_comb,
                )
                results.append(result)
            return results

        # Use joblib to parallelize the work
        max_nbytes = "1M" # max array size before memory mapping is triggered
        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"]))
        chunks = [range(i, min(i + chunk_size, nbins)) for i in range(0, nbins, chunk_size)]
        parallel_configs = {"n_jobs": config["GENERAL"]["NCPU"], "max_nbytes": max_nbytes, "temp_folder": memmap_folder, "mmap_mode": "c"}
        ppxf_tmp = Parallel(**parallel_configs)(delayed(worker)(chunk, templates) for chunk in chunks)

        # Flatten the results
        ppxf_tmp = [result for chunk_results in ppxf_tmp for result in chunk_results]

        # Unpack results
        for i in range(0, nbins):
            ppxf_result[i, : config["KIN"]["MOM"]] = ppxf_tmp[i][0]
            ppxf_reddening[i] = ppxf_tmp[i][1]
            ppxf_bestfit[i, :] = ppxf_tmp[i][2]
            optimal_template[i, :] = ppxf_tmp[i][3]
            mc_results[i, : config["KIN"]["MOM"]] = ppxf_tmp[i][4]
            formal_error[i, : config["KIN"]["MOM"]] = ppxf_tmp[i][5]
            spectral_mask[i, :] = ppxf_tmp[i][6]
            snr_postfit[i] = ppxf_tmp[i][7]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, nbins):
            # for i in range(1, 2):
            (
                ppxf_result[i, : config["KIN"]["MOM"]],
                ppxf_reddening[i],
                ppxf_bestfit[i, :],
                optimal_template[i, :],
                mc_results[i, : config["KIN"]["MOM"]],
                formal_error[i, : config["KIN"]["MOM"]],
                spectral_mask[i, :],
                snr_postfit[i],
            ) = run_ppxf(
                templates,
                bin_data[:, i],
                noise[:, i],
                velscale,
                start[i, :],
                goodPixels_ppxf,
                config["KIN"]["MOM"],
                config["KIN"]["ADEG"],
                config["KIN"]["MDEG"],
                config["KIN"]["REDDENING"],
                config["KIN"]["DOCLEAN"],
                logLam,
                offset,
                velscale_ratio,
                nsims,
                nbins,
                i,
                optimal_template_comb,
            )
        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print(
        "             Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )
    logging.info(
        "Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check for exceptions which occurred during the analysis
    idx_error = np.where(np.isnan(ppxf_result[:, 0]) == True)[0]
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

    # Save stellar kinematics to file
    save_ppxf(
        config,
        ppxf_result,
        ppxf_reddening,
        mc_results,
        formal_error,
        ppxf_bestfit,
        logLam,
        goodPixels_ppxf,
        optimal_template,
        logLam_template,
        npix,
        spectral_mask,
        optimal_template_comb,
        bin_data,
        snr_postfit,
    )

    # Return

    return None
