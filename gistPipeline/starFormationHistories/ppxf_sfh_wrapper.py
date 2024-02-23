import glob
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

# Physical constants
C = 299792.458  # speed of light in km/s


"""
PURPOSE:
  This module performs the extraction of non-parametric star-formation histories
  by full-spectral fitting.  Basically, it acts as an interface between pipeline
  and the pPXF routine from Cappellari & Emsellem 2004
  (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""

def robust_sigma(y, zero=False):
     """
     Biweight estimate of the scale (standard deviation).
     Implements the approach described in
     "Understanding Robust and Exploratory Data Analysis"
     Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417
     Added for sigma-clipping method
     """
     y = np.ravel(y)
     d = y if zero else y - np.median(y)

     mad = np.median(np.abs(d))
     u2 = (d/(9.0*mad))**2  # c = 9
     good = u2 < 1.0
     u1 = 1.0 - u2[good]
     num = y.size * ((d[good]*u1**2)**2).sum()
     den = (u1*(1.0 - 5.0*u2[good])).sum()
     sigma = np.sqrt(num/(den*(den - 1.0)))  # see note in above reference

     return sigma

def run_ppxf_firsttime(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    offset,
    degree,
    mdeg,
    regul_err,
    doclean,
    fixed,
    velscale_ratio,
    npix,
    ncomb,
    nbins,
    i,
    optimal_template_in,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
        # Call PPXF for first time to get optimal template
    #print("Running pPXF for the first time")
    #logging.info("Using the new 3-step pPXF implementation")
    pp = ppxf(
        templates,
        log_bin_data,
        log_bin_error,
        velscale,
        start,
        goodpixels=goodPixels,
        plot=False,
        quiet=True,
        moments=nmoments,
        degree=-1,
        vsyst=offset,
        mdegree=mdeg,
        regul = 1./regul_err,
        fixed=fixed,
        velscale_ratio=velscale_ratio,
    )

    normalized_weights = pp.weights / np.sum( pp.weights )
    optimal_template   = np.zeros( templates.shape[0] )
    for j in range(0, templates.shape[1]):
        optimal_template = optimal_template + templates[:,j]*normalized_weights[j]

    return optimal_template

def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    offset,
    degree,
    mdeg,
    regul_err,
    doclean,
    fixed,
    velscale_ratio,
    npix,
    ncomb,
    nbins,
    i,
    optimal_template_in,
):

    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    printStatus.progressBar(i, nbins, barLength=50)

    try:

        if len(optimal_template_in) > 1:
            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise=np.full_like(log_bin_data, 1.0)

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
                degree=-1,
                vsyst=offset,
                mdegree=mdeg,
                fixed=fixed,
                velscale_ratio=velscale_ratio,
            )
            # Find a proper estimate of the noise
            #noise_orig = biweight_location(log_bin_error[goodPixels])
            #goodpixels is one shorter than log_bin_error
            noise_orig = np.mean(log_bin_error[goodPixels])
            noise_est = robust_sigma(pp_step1.galaxy[goodPixels]-pp_step1.bestfit[goodPixels])

            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error*(noise_est/noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est-noise_new_std)] = noise_est

            ################ 2 ##################
            # Second Call PPXF - use best-fitting template, determine outliers
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
                    degree=-1,
                    vsyst=offset,
                    mdegree=mdeg,
                    fixed=fixed,
                    velscale_ratio=velscale_ratio,
                    clean=True,
                )

                # update goodpixels
                goodPixels = pp_step2.goodpixels

                # repeat noise scaling # Find a proper estimate of the noise
                noise_orig = biweight_location(log_bin_error[goodPixels])
                noise_est = robust_sigma(pp_step2.galaxy[goodPixels]-pp_step2.bestfit[goodPixels])

                # Calculate the new noise, and the sigma of the distribution.
                noise_new = log_bin_error*(noise_est/noise_orig)
                noise_new_std = robust_sigma(noise_new)

                # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
                noise_new[np.where(noise_new <= noise_est-noise_new_std)] = noise_est

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
                degree=-1,
                vsyst=offset,
                mdegree=mdeg,
                regul = 1./regul_err,
                fixed=fixed,
                velscale_ratio=velscale_ratio,
            )

        #update goodpixels again
        goodPixels = pp.goodpixels

        #make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # Calculate the true S/N from the residual
        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmean(pp.galaxy[goodPixels]/noise_est)

        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum( pp.weights )
        optimal_template   = np.zeros( templates.shape[0] )
        for j in range(0, templates.shape[1]):
            optimal_template = optimal_template + templates[:,j]*normalized_weights[j]

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum()
        w_row   = np.reshape(weights, ncomb)

        # # Do MC-Simulations - Amelia - this is not currently implemented. Add back in later.
        # sol_MC     = np.zeros((nsims,nmoments))
        mc_results = np.zeros(nmoments)
        #
        # for o in range(0, nsims):
        #     # Add noise to bestfit:
        #     #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
        #     #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
        #     noisy_bestfit = pp.bestfit  +  np.random.normal(0, 1, len(log_bin_data)) * np.std( log_bin_data[goodPixels] - pp.bestfit[goodPixels] )
        #
        #     mc = ppxf(templates, noisy_bestfit, log_bin_error, velscale, start, goodpixels=goodPixels, plot=False, \
        #             quiet=True, moments=nmoments, degree=-1, mdegree=mdeg, velscale_ratio=velscale_ratio, vsyst=offset, bias=0.0)
        #     sol_MC[o,:] = mc.sol[:]
        #
        # if nsims != 0:
        #     mc_results = np.nanstd( sol_MC, axis=0 )
        # print(pp.sol[:])

        return(
            pp.sol[:],
            w_row,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
        )

    except:
        return( np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)



# ## *****************************************************************************
# #        noise_i = noise_i * np.sqrt(  / len(goodPixels) )
# #        regul_err =
#
#         pp = ppxf(templates, galaxy_i, noise_i, velscale, start, goodpixels=goodPixels, plot=False, quiet=True,\
#               moments=nmom, degree=-1, vsyst=dv, mdegree=mdeg, regul=1./regul_err, fixed=fixed, velscale_ratio=velscale_ratio)
#
# #        if i == 0:
# #            print()
# #            print( i, pp.chi2 )
# #            print( len( goodPixels ) )
# #            print( np.sqrt(2 * len(goodPixels)) )
# #            print()
#
#         weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum()
#         w_row   = np.reshape(weights, ncomb)
#
#         # Correct the formal errors assuming that the fit is good
#         formal_error = pp.error * np.sqrt(pp.chi2)
#
#         return(pp.sol, w_row, pp.bestfit, formal_error)
#
#     except:
#         return(np.nan, np.nan, np.nan, np.nan)



def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin.
    """
    mean = np.zeros( (nbins,3) ); mean[:,:] = np.nan

    for i in range( nbins ):
        mean[i,0] = np.sum(w_row[i] * ageGrid.ravel())   / np.sum(w_row[i])
        mean[i,1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
        mean[i,2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])

    return(mean)


def save_sfh(
    mean_result,
    ppxf_result,
    w_row,
    mc_results,
    formal_error,
    logAge_grid,
    metal_grid,
    alpha_grid,
    ppxf_bestfit,
    logLam,
    goodPixels,
    velscale,
    logLam1,
    ncomb,
    nAges,
    nMetal,
    nAlpha,
    npix,
    config,
    spectral_mask,
    optimal_template_comb,
    snr_postfit,
):
    """ Save all results to disk. """

    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    # Define the columns
    columns = [
        fits.Column(name="AGE", format="D", array=mean_result[:, 0]),
        fits.Column(name="METAL", format="D", array=mean_result[:, 1]),
        fits.Column(name="ALPHA", format="D", array=mean_result[:, 2]),
        fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:])
    ]

    if not config["SFH"]["FIXED"]:
        columns.extend([
            fits.Column(name="V", format="D", array=kin[:, 0]),
            fits.Column(name="SIGMA", format="D", array=kin[:, 1]),
            fits.Column(name="FORM_ERR_V", format="D", array=formal_error[:, 0]),
            fits.Column(name="FORM_ERR_SIGMA", format="D", array=formal_error[:, 1])
        ])

        for i in range(2, 6):
            if np.any(kin[:, i]) != 0:
                columns.append(fits.Column(name=f"H{i+1}", format="D", array=kin[:, i]))
            if np.any(formal_error[:, i]) != 0:
                columns.append(fits.Column(name=f"FORM_ERR_H{i+1}", format="D", array=formal_error[:, i]))

    # Create the HDUs
    priHDU = fits.PrimaryHDU()
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="SFH")

    # Save the configuration to the headers
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE WEIGHTS AND GRID
    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh-weights.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with weights
    cols_weights = [fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_weights), name="WEIGHTS")

    # Reshape the grids
    logAge_row, metal_row, alpha_row = map(np.reshape, [logAge_grid, metal_grid, alpha_grid], [ncomb]*3)

    # Table HDU with grids
    cols_grid = [fits.Column(name=name, format="D", array=array) 
                 for name, array in zip(["LOGAGE", "METAL", "ALPHA"], [logAge_row, metal_row, alpha_row])]
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_grid), name="GRID")

    # Create HDU list and write to file
    HDUList = fits.HDUList([_auxiliary.saveConfigToHeader(hdu, config["SFH"]) for hdu in [priHDU, dataHDU, gridHDU]])
    HDUList.writeto(outfits_sfh, overwrite=True)

    # Set additional header values
    for name, value in zip(["NAGES", "NMETAL", "NALPHA"], [nAges, nMetal, nAlpha]):
        fits.setval(outfits_sfh, name, value=value)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits"
    )
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE BESTFIT
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh-bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-bestfit.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with SFH bestfit
    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=ppxf_bestfit ))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Table HDU with SFH logLam
    cols = []

    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))

    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Table HDU with SFH goodpixels
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["SFH"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh, "VELSCALE", value=velscale)
    fits.setval(outfits_sfh, "CRPIX1", value=1.0)
    fits.setval(outfits_sfh, "CRVAL1", value=logLam1[0])
    fits.setval(outfits_sfh, "CDELT1", value=logLam1[1] - logLam1[0])

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-bestfit.fits"
    )
    logging.info("Wrote: " + outfits_sfh)


def extractStarFormationHistories(config):
    """
    Starts the computation of non-parametric star-formation histories with
    pPXF.  A spectral template library sorted in a three-dimensional grid of
    age, metallicity, and alpha-enhancement is loaded.  Emission-subtracted
    spectra are used for the fit. An according emission-line mask is
    constructed. The stellar kinematics can or cannot be fixed to those obtained
    with a run of unregularized pPXF and the analysis started.  Results are
    saved to disk and the plotting routines called.
    Args:
    - config: dictionary containing configuration parameters
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    # Prepare template library
    
    # Open the HDF5 file
    with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5", 'r') as f:
        # Read the VELSCALE attribute from the file
        velscale = f.attrs['VELSCALE']
        
    velscale_ratio = 2

    (
        templates,
        lamRange_temp,
        logLam_template,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        config['SFH']['LMIN'],
        config['SFH']['LMAX'],
        velscale/velscale_ratio,
        LSF_Data,
        LSF_Templates,
        'SFH',
        sortInGrid=True,
    )
    templates = templates.reshape( (templates.shape[0], ntemplates) )

    # Define file paths
    gas_cleaned_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + '_gas-cleaned_'+config['GAS']['LEVEL']+'.fits'
    bin_spectra_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5"

    # Check if emission-subtracted spectra file exists
    if os.path.isfile(gas_cleaned_file):
        logging.info(f"Using emission-subtracted spectra at {gas_cleaned_file}")
        printStatus.done("Using emission-subtracted spectra")
        # Open the FITS file
        with fits.open(gas_cleaned_file, mem_map=True) as hdul:
            # Read the LOGLAM data from the file
            logLam = hdul[2].data['LOGLAM']

            # Select the indices where the wavelength is within the specified range
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config['SFH']['LMIN'], np.exp(logLam) < config['SFH']['LMAX']))[0]

            # Read the SPEC and ESPEC data from the file, only for the selected indices
            galaxy = hdul[1].data['SPEC'][:, idx_lam]
            bin_data = hdul[1].data['SPEC'].T[idx_lam, :]
            bin_err = hdul[1].data['ESPEC'].T[idx_lam, :]
            logLam = logLam[idx_lam]
    else:
        logging.info(f"Using regular spectra without any emission-correction at {bin_spectra_file}")
        printStatus.done("Using regular spectra without any emission-correction")
        with h5py.File(bin_spectra_file, 'r') as f:
            # Read the LOGLAM data from the file
            logLam = f['LOGLAM'][:]

            # Select the indices where the wavelength is within the specified range
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config['SFH']['LMIN'], np.exp(logLam) < config['SFH']['LMAX']))[0]

            # Read the SPEC and ESPEC data from the file, only for the selected indices
            galaxy = f['SPEC'][:, idx_lam].T
            bin_data = f['SPEC'][idx_lam, :]
            bin_err = f['ESPEC'][idx_lam, :]
            logLam = logLam[idx_lam]

    # Define additional variables
    nbins = galaxy.shape[0]
    npix = galaxy.shape[1]
    ubins = np.arange(nbins)
    noise = np.full(npix, config['SFH']['NOISE'])
    dv = (np.log(lamRange_temp[0]) - logLam[0])*C

    # Apply the selection to the logLam array


    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    noise = bin_err  # is actual noise, not variance
    nsims = config['SFH']['MC_PPXF']


    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config["SFH"]["FIXED"] == True:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        # Set fixed option to True
        fixed = [True] * config["KIN"]["MOM"]

        # Read PPXF results
        ppxf_data = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True
        )[1].data
        start = np.zeros((nbins, config["KIN"]["MOM"]))
        for i in range(nbins):
            start[i, :] = np.array(ppxf_data[i][: config["KIN"]["MOM"]])

    # Do *NOT* fix kinematics to those obtained previously
    elif config["SFH"]["FIXED"] == False:
        logging.info(
            "Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties."
        )
        # Set fixed option to False and use initial guess from Config-file
        fixed = None
        start = np.zeros((nbins, 2))
        for i in range(nbins):
            start[i, :] = np.array([0.0, config["KIN"]["SIGMA"]])

    # Define goodpixels
    goodPixels_sfh = _auxiliary.spectralMasking(config, config['SFH']['SPEC_MASK'], logLam)

    # Define output arrays
    ppxf_result = np.zeros((nbins,6    ))
    w_row = np.zeros((nbins,ncomb))
    ppxf_bestfit = np.zeros((nbins,npix))
    optimal_template = np.zeros((nbins,templates.shape[0]))
    mc_results = np.zeros((nbins,6))
    formal_error = np.zeros((nbins,6))
    spectral_mask = np.zeros((nbins,bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)

    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:,:],axis=1)
    comb_espec = np.nanmean(bin_err[:,:],axis=1)
    #comb_spec = comb_spec/np.nanmedian(comb_spec) # Amelia added to mormalise normalize spectrum
    #comb_espec = comb_espec/np.nanmedian(comb_espec) # and the error spectrum
    optimal_template_init = [0]

    optimal_template_out = run_ppxf_firsttime(
        templates,
        comb_spec ,
        comb_espec,
        velscale,
        start[0,:],
        goodPixels_sfh,
        config['SFH']['MOM'],
        offset,-1,
        config['SFH']['MDEG'],
        config['SFH']['REGUL_ERR'],
        config["SFH"]["DOCLEAN"],
        fixed,
        velscale_ratio,
        ncomb,
        nsims,
        nbins,
        0,
        optimal_template_init,
    )

    # now define the optimal template that we'll use throughout
    optimal_template_comb = optimal_template_out

    # ====================


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
                    bin_data[:,i],
                    noise[:,i],
                    velscale,
                    start[i,:],
                    goodPixels_sfh,
                    config['SFH']['MOM'],
                    offset,
                    -1,
                    config['SFH']['MDEG'],
                    config['SFH']['REGUL_ERR'],
                    config["SFH"]["DOCLEAN"],
                    fixed,
                    velscale_ratio,
                    npix,
                    ncomb,
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
            ppxf_result[i,:config['SFH']['MOM']] = ppxf_tmp[i][0]
            w_row[i,:] = ppxf_tmp[i][1]
            ppxf_bestfit[i,:] = ppxf_tmp[i][2]
            optimal_template[i,:] = ppxf_tmp[i][3]
            mc_results[i,:config['SFH']['MOM']] = ppxf_tmp[i][4]
            formal_error[i,:config['SFH']['MOM']] = ppxf_tmp[i][5]
            spectral_mask[i,:] = ppxf_tmp[i][6]
            snr_postfit[i] = ppxf_tmp[i][7]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False: # Amelia you haven't tested this yet. Come back to.
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(nbins):
            (
                kin[i,:config['SFH']['MOM']],
                w_row[i,:],
                bestfit[i,:],
                formal_error[i,:config['SFH']['MOM']],
                snr_postfit[i],
            ) = run_ppxf(
                templates,
                galaxy[i,:],
                noise,
                velscale,
                start[i,:],
                goodPixels_sfh,
                config['SFH']['MOM'],
                dv,
                -1,
                config['SFH']['MDEG'],
                config['SFH']['REGUL_ERR'],
                config["KIN"]["DOCLEAN"],
                fixed,
                velscale_ratio,
                npix,
                ncomb,
                nbins,
                i,
                optimal_template_init,
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
    idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]

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

    # Calculate mean age, metallicity and alpha
    mean_results = mean_agemetalalpha(
        w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins
    )

    # Save to file

    save_sfh(
        mean_results,
        ppxf_result,
        w_row,
        mc_results,
        formal_error,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ppxf_bestfit,
        logLam,
        goodPixels_sfh,
        velscale,
        logLam,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
        npix,
        config,
        spectral_mask,
        optimal_template_comb,
        snr_postfit,
    )

    # Return
    return None
