import glob
import logging
import os
import time

import h5py
import numpy as np
from astropy import table
from astropy.io import fits
from joblib import Parallel, delayed, dump, load
# Then use system installed version instead
from ppxf.ppxf import ppxf
from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary
from gistPipeline.prepareTemplates import (_prepareTemplates,
                                           prepare_gas_templates)

# Physical constants
C = 299792.458  # speed of light in km/s


"""
PURPOSE:
  This module executes the emission-line analysis of the pipeline.
  Uses the pPXF implementation, replacing Gandalf.
  Module written for gist-geckos based on the gist SFH module
  combined with the PHANGS DAP emission line module.
"""

def run_ppxf(
    templates,
    galaxy_i,
    noise_i,
    velscale,
    start,
    goodPixels,
    tpl_comp,
    moments,
    offset,
    mdeg,
    fixed,
    velscale_ratio,
    tied,
    gas_comp,
    gas_names,
    i,
    nbins,
    ubins,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    printStatus.progressBar(i, np.max(ubins) + 1, barLength=50)

    try:
        pp = ppxf(
            templates,
            galaxy_i,
            noise_i,
            velscale,
            start,
            goodpixels=goodPixels,
            plot=False,
            quiet=True,
            component=tpl_comp,
            moments=moments,
            degree=-1,
            vsyst=offset,
            mdegree=mdeg,
            fixed=fixed,
            velscale_ratio=velscale_ratio,
            tied=tied,
            gas_component=gas_comp,
            gas_names=gas_names,
        )

        return (
            pp.sol[1:],
            pp.error[1:],
            pp.chi2,
            pp.gas_flux,
            pp.gas_flux_error,
            pp.gas_names,
            pp.bestfit,
            pp.gas_bestfit,
            pp.sol[0],
            pp.error[0],
        )

    except:
        # raise exception
        printStatus.running("Emission line fit failed")
        logging.info("Emission line fit failed")

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )


def tidy_up_fluxes_and_kinematics(
    gas_kinematics,
    kinematics_all_err,
    gas_flux,
    gas_flux_error,
    emldb,
    eml_tying,
    gas_moments,
):
    """
    Function taken from util_templates.py of PHANGS DAP. Tidies up some stuff.
    """
    w = emldb["action"] == "f"
    linesfitted = emldb[w]
    tpli_fitted = eml_tying["tpli"][w]
    nbins = gas_kinematics.shape[0]
    ngascomp = len(np.unique(eml_tying["comp"]))
    ngastpl = len(np.unique(tpli_fitted))

    fluxes_final = np.zeros((nbins, len(tpli_fitted)))
    fluxes_err_final = np.zeros((nbins, len(tpli_fitted)))
    vel_final = np.zeros((nbins, len(tpli_fitted)))
    vel_err_final = np.zeros((nbins, len(tpli_fitted)))
    sigma_final = np.zeros((nbins, len(tpli_fitted)))
    sigma_err_final = np.zeros((nbins, len(tpli_fitted)))
    if gas_moments >= 3:
        h3_final = np.zeros((nbins, len(tpli_fitted)))
    if gas_moments == 4:
        h4_final = np.zeros((nbins, len(tpli_fitted)))

    for j in range(ngastpl):
        # which line(s) correspond to this template
        templates_x = np.arange(len(tpli_fitted))[tpli_fitted == j]
        # component number of this template
        component_x = eml_tying["comp"][j]
        # if more than one line in this template...
        if len(templates_x) > 1:
            total_flux_template = np.sum(linesfitted["A_i"][templates_x])
            stronger_line = templates_x[linesfitted["A_i"][templates_x] == 1.0][0]
            weaker_lines = templates_x[linesfitted["A_i"][templates_x] != 1.0]
            fluxes_final[:, stronger_line] = gas_flux[:, j] / total_flux_template
            fluxes_err_final[:, stronger_line] = (
                gas_flux_error[:, j] / total_flux_template
            )

            for kk in range(len(weaker_lines)):
                fluxes_final[:, weaker_lines[kk]] = (
                    gas_flux[:, j]
                    / total_flux_template
                    * linesfitted["A_i"][weaker_lines[kk]]
                )
                fluxes_err_final[:, weaker_lines[kk]] = (
                    gas_flux_error[:, j]
                    / total_flux_template
                    * linesfitted["A_i"][weaker_lines[kk]]
                )

        else:
            fluxes_final[:, templates_x[0]] = gas_flux[:, j]
            fluxes_err_final[:, templates_x[0]] = gas_flux_error[:, j]

    for j in range(ngastpl):
        # which line(s) correspond to this template
        # linesfitted[templates_x] will give you the info on this line(s)
        templates_x = np.arange(len(tpli_fitted))[tpli_fitted == j]
        # component number of this template
        component_x = eml_tying["comp"][j]
        # now list all the components that can have
        # sigma and velocity group number of this template
        sigma_x = eml_tying["comp"][eml_tying["sgrp"] == eml_tying["sgrp"][j]]
        v_x = eml_tying["comp"][eml_tying["vgrp"] == eml_tying["vgrp"][j]]

        # in case there is a component with a lower number than component_x
        # in this velocity/sigma group it means the  kinematics of that line
        # is tied to the one of that component, so replace  component_x with component_vx
        # in order to get the ERRORS of the primary (aka free) component
        # otherwise you would get zerro errors because line is tied
        component_vx = np.min(v_x)
        component_sx = np.min(sigma_x)

        # components tied which have therefore zero errors
        # THESE LINES causes ENDLESS problems because of the ==0 conditions
        # tied_vel  = np.arange(ngascomp)[kinematics_all_err[0, :, 0]==0]
        # tied_sigma  = np.arange(ngascomp)[kinematics_all_err[0, :, 1]==0]

        #
        # if component_x in tied_vel:
        #     component_vx = v_x[np.array([i not in tied_vel for i in v_x])][0]
        # if component_x in tied_sigma:
        #     component_sx = sigma_x[np.array([i not in tied_sigma for i in sigma_x])][0]

        for kk in range(len(templates_x)):
            vel_final[:, templates_x[kk]] = gas_kinematics[:, component_x, 0]
            sigma_final[:, templates_x[kk]] = gas_kinematics[:, component_x, 1]

            vel_err_final[:, templates_x[kk]] = kinematics_all_err[:, component_vx, 0]
            sigma_err_final[:, templates_x[kk]] = kinematics_all_err[:, component_sx, 1]

            extra = None
            if gas_moments >= 3:
                h3_final[:, templates_x[kk]] = gas_kinematics[:, component_x, 2]
                extra = {}
                extra["h3"] = h3_final
            if gas_moments == 4:
                h4_final[:, templates_x[kk]] = gas_kinematics[:, component_x, 3]
                extra["h4"] = h4_final

    return (
        linesfitted,
        fluxes_final,
        fluxes_err_final,
        vel_final,
        vel_err_final,
        sigma_final,
        sigma_err_final,
        extra,
    )


def save_ppxf_emlines(
    config,
    rootname,
    outdir,
    level,
    linesfitted,
    gas_flux_in_units,
    gas_err_flux_in_units,
    vel_final,
    vel_err_final,
    sigma_final_measured,
    sigma_err_final,
    chi2,
    templates_sigma,
    bestfit,
    gas_bestfit,
    stkin,
    spectra,
    error,
    goodPixels_gas,
    logLam_galaxy,
    ubins,
    npix,
    extra,
):
    # ========================
    # SAVE RESULTS
    outfits_ppxf = rootname + "/" + outdir + "_gas_" + level + ".fits"
    printStatus.running("Writing: " + outfits_ppxf.split("/")[-1])
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_gas_" + level + ".fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF output data
    cols = []
    cols.append(fits.Column(name="BIN_ID", format="J", array=ubins))
    cols.append(fits.Column(name="V_STARS2", format="D", array=stkin[:, 0]))
    cols.append(fits.Column(name="SIGMA_STARS2", format="D", array=stkin[:, 1]))
    cols.append(fits.Column(name="CHI2_TOT", format="D", array=chi2))
    # emission lines names
    names = np.char.array(linesfitted["name"]) + np.char.array(
        ["{:d}".format(int(j)) for j in linesfitted["lambda"]]
    )

    for i in range(len(names)):
        cols.append(
            fits.Column(
                name=names[i] + "_FLUX", format="D", array=gas_flux_in_units[:, i]
            )
        )
        cols.append(
            fits.Column(
                name=names[i] + "_FLUX_ERR",
                format="D",
                array=gas_err_flux_in_units[:, i],
            )
        )
        cols.append(
            fits.Column(name=names[i] + "_VEL", format="D", array=vel_final[:, i])
        )
        cols.append(
            fits.Column(
                name=names[i] + "_VEL_ERR", format="D", array=vel_err_final[:, i]
            )
        )
        cols.append(
            fits.Column(
                name=names[i] + "_SIGMA", format="D", array=sigma_final_measured[:, i]
            )
        )
        cols.append(
            fits.Column(
                name=names[i] + "_SIGMA_ERR", format="D", array=sigma_err_final[:, i]
            )
        )
        cols.append(
            fits.Column(
                name=names[i] + "_SIGMA_CORR",
                format="D",
                array=np.zeros(sigma_final_measured.shape[0]) + templates_sigma[i],
            )
        )

        if (extra is not None) and ("h3" in extra.keys()):
            cols.append(
                fits.Column(name=names[i] + "_H3", format="D", array=extra["h3"][:, i])
            )
        if (extra is not None) and ("h4" in extra.keys()):
            cols.append(
                fits.Column(name=names[i] + "_H4", format="D", array=extra["h4"][:, i])
            )

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "EMISSION"

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.running("Writing: " + outfits_ppxf.split("/")[-1])
    logging.info("Wrote: " + outfits_ppxf)

    # ========================
    # SAVE CLEANED SPECTRUM
    outfits_ppxf = rootname + "/" + outdir + "_gas-cleaned_" + level + ".fits"
    printStatus.running("Writing: " + outfits_ppxf.split("/")[-1])
    cleaned = spectra.T - gas_bestfit
    spec = spectra.T
    err = error.T

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with cleaned spectrum
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=cleaned))
    cols.append(fits.Column(name="ESPEC", format=str(npix) + "D", array=err))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "CLEANED_SPECTRA"

    # Extension 2: Table HDU with logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam_galaxy))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Extension 3: Table HDU with bestfit
    cols = []
    cols.append(fits.Column(name="BESTFIT", format=str(npix) + "D", array=bestfit))
    bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    bestfitHDU.name = "BESTFIT"

    # Extension 3: Table HDU with gas bestfit
    cols = []
    cols.append(
        fits.Column(name="GAS_BESTFIT", format=str(npix) + "D", array=gas_bestfit)
    )
    gas_bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gas_bestfitHDU.name = "GAS_BESTFIT"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["GAS"])
    bestfitHDU = _auxiliary.saveConfigToHeader(bestfitHDU, config["GAS"])
    gas_bestfitHDU = _auxiliary.saveConfigToHeader(gas_bestfitHDU, config["GAS"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, bestfitHDU, gas_bestfitHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    # ========================
    # SAVE BESTFIT
    outfits_ppxf = rootname + "/" + outdir + "_gas-bestfit_" + level + ".fits"
    printStatus.running("Writing: " + outfits_ppxf.split("/")[-1])
    # cleaned = spectra.T - gas_bestfit
    spec = spectra.T
    err = error.T

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with cleaned spectrum
    cols = []
    cols.append(fits.Column(name="BESTFIT", format=str(npix) + "D", array=bestfit))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Extension 2: Table HDU with logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam_galaxy))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Extension 3: Table HDU with bestfit
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels_gas))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Extension 3: Table HDU with gas bestfit
    cols = []
    cols.append(
        fits.Column(name="GAS_BESTFIT", format=str(npix) + "D", array=gas_bestfit)
    )
    gas_bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gas_bestfitHDU.name = "GAS_BESTFIT"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["GAS"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["GAS"])
    gas_bestfitHDU = _auxiliary.saveConfigToHeader(gas_bestfitHDU, config["GAS"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU, gas_bestfitHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.running("Writing: " + outfits_ppxf.split("/")[-1])
    logging.info("Wrote: " + outfits_ppxf)


def performEmissionLineAnalysis(config):  # This is your main emission line fitting loop
    # print("")
    # print("\033[0;37m"+" - - - - - Running Emission Lines Fitting - - - - - "+"\033[0;39m")
    logging.info(" - - - Running Emission Lines Fitting - - - ")

    # #--> some bookkeeping - all commented out for now Amelia
    # # if there is only one spectrum to fit, make sure to reformat it
    # if log_spec.ndim==1:
    #     log_error= np.expand_dims(log_error, axis=1)
    #     log_spec= np.expand_dims(log_spec, axis=1)
    # nbins    = log_spec.shape[1]
    # ubins    = np.arange(0, nbins)
    # npix_in     = log_spec.shape[0]
    # n_spaxels_per_bin = np.zeros(nbins)
    # if bin_id is None:
    #     bin_id = ubins
    # # number of spaxels per bin
    # for i in range(nbins):
    #     windx = (bin_id ==i)
    #     n_spaxels_per_bin[i]=np.sum(windx)
    # velscale_ratio = 1
    # # check if wavelength is in vacuum
    # if 'WAV_VACUUM' in configs:
    #     wav_in_vacuum = configs['WAV_VACUUM']
    # else:
    #     wav_in_vacuum = False

    # for now the number of gas moments is fixed to 2 (i.e. v and sigma, no h3 and h4 etc for gas)

    ## --------------------- ##
    # For output filenames
    if config["GAS"]["LEVEL"] in ["BIN", "SPAXEL"]:
        currentLevel = config["GAS"]["LEVEL"]
    elif (
        config["GAS"]["LEVEL"] == "BOTH"
        and os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas_BIN.fits"  # If you haven't already created the BIN products
        )
        == False
    ):
        currentLevel = "BIN"
    elif (
        config["GAS"]["LEVEL"] == "BOTH"
        and os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas_BIN.fits"  # If you have created BIN products, move on to SPAXEL
        )
        == True
    ):
        currentLevel = "SPAXEL"
        print("currentLevel = %s" % (currentLevel))
    # Oversample the templates by a factor of two
    velscale_ratio = 2

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "GAS")
    ## --------------------- ##

    emi_mpol_deg = config["GAS"]["MDEG"]  # Should be ~8
    ## --------------------- ##
    # Read data if we run on BIN level
    if currentLevel == "BIN":
        # Open the HDF5 file
        with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5", 'r') as f:
            # Read the data from the file
            spectra = f['SPEC'][:]
            error = f['ESPEC'][:]
            logLam_galaxy = f['LOGLAM'][:]
            velscale = f.attrs['VELSCALE']

        # Select the indices where the wavelength is within the specified range
        idx_lam = np.where(
            np.logical_and(
                np.exp(logLam_galaxy) > config["GAS"]["LMIN"],
                np.exp(logLam_galaxy) < config["GAS"]["LMAX"],
            )
        )[0]

        # Apply the selection to the spectra, error, and logLam_galaxy arrays
        spectra = spectra[idx_lam, :]
        error = error[idx_lam, :]
        logLam_galaxy = logLam_galaxy[idx_lam]

        npix = spectra.shape[0]
        nbins = spectra.shape[1]
        ubins = np.arange(0, nbins)
        nstmom = config["KIN"]["MOM"]  # Usually = 4
        # # the wav range of the data (observed)
        LamRange = (np.exp(logLam_galaxy[0]), np.exp(logLam_galaxy[-1]))

        # Determining the number of spaxels per bin
        hdu2 = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_table.fits", mem_map=True
        )
        bin_id = hdu2["TABLE"].data["BIN_ID"]
        n_spaxels_per_bin = np.zeros(nbins)
        if bin_id is None:
            bin_id = ubins
        # number of spaxels per bin
        for i in range(nbins):
            windx = bin_id == i
            n_spaxels_per_bin[i] = np.sum(windx)

        # Prepare templates - This is for the stellar templates
        logging.info("Using full spectral library for ppxf on BIN level")
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
        star_templates = templates.reshape((templates.shape[0], n_templates))

        offset = (logLam_template[0] - logLam_galaxy[0]) * C  # km/s
        # error        = np.ones((npix,nbins))
        ## --------------------- ##

    if currentLevel == "SPAXEL":

        # Open the HDF5 file
        with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_AllSpectra.hdf5", 'r') as f:
            # Read the data from the file
            spectra = f['SPEC'][:]
            error = f['ESPEC'][:]
            logLam_galaxy = f['LOGLAM'][:]
            velscale = f.attrs['VELSCALE']

        # Select the indices where the wavelength is within the specified range
        idx_lam = np.where(
            np.logical_and(
                np.exp(logLam_galaxy) > config["GAS"]["LMIN"],
                np.exp(logLam_galaxy) < config["GAS"]["LMAX"],
            )
        )[0]

        # Apply the selection to the spectra, error, and logLam_galaxy arrays
        spectra = spectra[idx_lam, :]
        error = error[idx_lam, :]
        logLam_galaxy = logLam_galaxy[idx_lam]

        npix = spectra.shape[0]
        nbins = spectra.shape[1]  # This should now = the number of spaxels
        ubins = np.arange(0, nbins)
        nstmom = config["KIN"]["MOM"]  # Usually = 4
        # # the wav range of the data (observed)
        LamRange = (np.exp(logLam_galaxy[0]), np.exp(logLam_galaxy[-1]))

        # Determining the number of spaxels per bin
        hdu2 = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_table.fits", mem_map=True
        )
        bin_id = hdu2["TABLE"].data["BIN_ID"]
        n_spaxels_per_bin = np.zeros(nbins)
        bin_id = ubins
        # number of spaxels per bin
        for i in range(nbins):  # Amelia - I dunno about this part - check tomorrow
            windx = ubins == i
            n_spaxels_per_bin[i] = np.sum(
                windx
            )  # This should be an array of ones, so useless?

        # Prepare templates - This is for the stellar templates
        logging.info("Using full spectral library for ppxf on SPAXEL level")
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
        star_templates = templates.reshape((templates.shape[0], n_templates))

        offset = (logLam_template[0] - logLam_galaxy[0]) * C  # km/s

    # --> generate the gas templates
    # emldb=table.Table.read('./configFiles/'+config['GAS']['EMI_FILE'] , format='ascii') # Now using the PHANGS emission line config file. NB change '/configFiles' to dirPath or something like that
    emldb = table.Table.read(
        config["GENERAL"]["CONFIG_DIR"] + "/" + config["GAS"]["EMI_FILE"],
        format="ascii",
    )  # Now using the PHANGS emission line config file. NB change '/configFiles' to dirPath or something like that

    # if wav_in_vacuum: # I dunno if we need this - will it ever be in a vaccumm?
    #     emldb['lambda'] = air_to_vac(emldb['lambda'])
    eml_fwhm_angstr = LSF_Templates(emldb["lambda"])
    # note that while the stellar templates are expanded in wavelength to cover +/- 150 Angstrom around the observed spectra (buffer)
    # emission line tempaltes are only generated for lines whose central wavelength lies within the min and max rest-frame waveelngth of the data
    (
        gas_templates,
        gas_names,
        line_wave,
        eml_tying,
    ) = prepare_gas_templates.generate_emission_lines_templates(
        emldb, LamRange, config, logLam_template, eml_fwhm_angstr
    )
    ngastpl = gas_templates.shape[1]

    # --> stack vertically stellar and gas templates
    templates = np.column_stack([star_templates, gas_templates])
    # New stuff that you need later that has come from util_templates.py  ine_emission_line_input_for_ppxf
    tpl_comp = np.append(
        np.zeros(star_templates.shape[1], dtype=int), eml_tying["comp"] + 1
    )
    # total number of templates
    n_templ = len(tpl_comp)
    # total number of (kinematic) components
    n_comp = len(np.unique(tpl_comp))
    # select gas components
    gas_comp = tpl_comp > 0
    # two moments per kinematics component
    moments = np.ones(n_comp, dtype=int) + 1
    moments[0] = config["KIN"]["MOM"]
    moments[1:] = 2  # gas moments hardcoded for now
    # total number of moments
    n_tot_moments = np.sum(moments)
    #      Parse the velocity and sigma groups into tied parameters
    tied_flat = np.empty(n_tot_moments, dtype=object)
    tpl_index = np.arange(n_templ)

    tpl_vgrp = np.append(
        np.zeros(star_templates.shape[1], dtype=int), eml_tying["vgrp"] + 1
    )
    tpl_sgrp = np.append(
        np.zeros(star_templates.shape[1], dtype=int), eml_tying["sgrp"] + 1
    )

    for i in range(n_comp):
        # Do not allow tying to fixed components?
        if moments[i] < 0:
            continue
        # Velocity group of this component
        indx = np.unique(tpl_comp[tpl_index[tpl_vgrp == i]])
        if len(indx) > 1:
            parn = [0 + np.sum(np.absolute(moments[:j])) for j in indx]
            tied_flat[parn[1:]] = "p[{0}]".format(parn[0])

        # Sigma group of this component
        indx = np.unique(tpl_comp[tpl_index[tpl_sgrp == i]])
        if len(indx) > 1:
            parn = [1 + np.sum(np.absolute(moments[:j])) for j in indx]
            tied_flat[parn[1:]] = "p[{0}]".format(parn[0])

    tied_flat[[t is None for t in tied_flat]] = ""

    # reshape the tied array so it matches the shape of start
    tied = []
    track = 0
    for i in range(n_comp):
        tied.append(list(tied_flat[track : track + np.absolute(moments[i])]))
        track = track + np.absolute(moments[i])

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config["GAS"]["FIXED"] == True:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        # Set fixed option to True
        fixed = [True] * config["KIN"]["MOM"]

        # Read PPXF results, add not just stellar, but 3x gas guesses
        ppxf_data = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True
        )[1].data
        # if config['GAS']['LEVEL'] == 'BIN':
        # No need to do anything!

        if currentLevel == "SPAXEL":
            binNum_long = np.array(
                fits.open(
                    os.path.join(
                        config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]
                    )
                    + "_table.fits", mem_map=True
                )[1].data.BIN_ID
            )
            ppxf_data_spaxels = np.zeros((len(ubins), len(ppxf_data[0])))
            nbins = np.max(binNum_long) + 1
            for i in range(int(nbins)):
                windx = binNum_long == i
                ppxf_data_spaxels[windx, :] = ppxf_data[i]
            ppxf_data = ppxf_data_spaxels

        # start = [[np.zeros((nbins, config['KIN']['MOM']))]] # old
        start, fixed = [], []
        for i in range(
            0, np.max(ubins) + 1
        ):  # what is nbins here? AMELIA up to here. Check this works for the spaxel case
            # start[i,:] = np.array( ppxf_data[i][:config['KIN']['MOM']] ) # old one (needs to be an array?)
            s = [
                ppxf_data[i][: config["KIN"]["MOM"]],
                [ppxf_data[i][0], 50],
                [ppxf_data[i][0], 50],
                [ppxf_data[i][0], 50],
            ]  # Here I am setting the starting stellar kin guess to the stellar kin results, and the starting gas vel guess to the stellar vel and starting gas sigma to 50
            start.append(s)
            f = [
                [1, 1, 1, 1],
                [0, 0],
                [0, 0],
                [0, 0],
            ]  # Fix the stellar kinematics, but not the gas
            fixed.append(f)

    # Do *NOT* fix kinematics to those obtained previously
    elif config["GAS"]["FIXED"] == False:
        logging.info(
            "Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties."
        )
        # Set fixed option to False and use initial guess from Config-file
        fixed = None
        start = np.zeros((nbins, 2))
        start, fixed = [], []
        for i in range(0, np.max(ubins) + 1):
            # start[i,:] = np.array( [0.0, config['SFH']['SIGMA']] ) # old
            s = [
                [0, config["KIN"]["SIGMA"]],
                [0, 50],
                [0, 50],
                [0, 50],
            ]  # Here the velocity guesses are all zero, the sigma guess is the stell kins sig guess for stars and 50 for the gas
            start.append(s)
            f = [
                [0, 0, 0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]  # Don't fix any of the kinematics because we're fitting them all!
            fixed.append(f)

    # Define goodpixels !
    goodPixels_gas = _auxiliary.spectralMasking(
        config, config["GAS"]["SPEC_MASK"], logLam_galaxy
    )

    n_gas_comp = 3  # len(np.unique(tpl_comp[gas_comp]))
    n_gas_templates = ngastpl  # len(tpl_comp[gas_comp]) ngastpl defined above

    # Array to store results of ppxf - come back to this, I don' think it's necc what I want?
    gas_kinematics = (
        np.zeros((np.max(ubins) + 1, n_gas_comp, 2)) + np.nan
    )  # the 2 was configs['GAS_MOMENTS'], but I think we just want 2 for now
    gas_kinematics_err = np.zeros((np.max(ubins) + 1, n_gas_comp, 2)) + np.nan
    chi2 = np.zeros((np.max(ubins) + 1))
    gas_flux = np.zeros((np.max(ubins) + 1, n_gas_templates))
    gas_flux_error = np.zeros((np.max(ubins) + 1, n_gas_templates))
    gas_named = []
    bestfit = np.zeros((np.max(ubins) + 1, npix))
    gas_bestfit = np.zeros((np.max(ubins) + 1, npix))
    stkin = np.zeros((np.max(ubins) + 1, nstmom))
    stkin_err = np.zeros((np.max(ubins) + 1, nstmom))

    # ====================
    # Run PPXF
    start_time = time.time()

    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF for emission lines analysis in parallel mode")
        logging.info("Running PPXF for emission lines analysis in parallel mode")

        # Prepare the folder where the memmap will be dumped
        memmap_folder = "/scratch" if os.access("/scratch", os.W_OK) else config["GENERAL"]["OUTPUT"]

        # dump the arrays and load as memmap
        templates_filename_memmap = memmap_folder + "/templates_memmap.tmp"
        dump(templates, templates_filename_memmap)
        templates = load(templates_filename_memmap, mmap_mode='r')
        
        spectra_filename_memmap = memmap_folder + "/spectra_memmap.tmp"
        dump(spectra, spectra_filename_memmap)
        spectra = load(spectra_filename_memmap, mmap_mode='r')
        
        error_filename_memmap = memmap_folder + "/error_memmap.tmp"
        dump(error, error_filename_memmap)
        error = load(error_filename_memmap, mmap_mode='r')

        # Define a function to encapsulate the work done in the loop
        def worker(chunk, templates):
            results = []
            for i in chunk:
                result = run_ppxf(
                    templates,
                    spectra[:, i],
                    error[:, i],
                    velscale,
                    start[i],
                    goodPixels_gas,
                    tpl_comp,
                    moments,
                    offset,
                    emi_mpol_deg,
                    fixed[i],
                    velscale_ratio,
                    tied,
                    gas_comp,
                    gas_names,
                    i,
                    nbins,
                    ubins,
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

        for i in range(0, nbins):
            gas_kinematics[i, :, :] = ppxf_tmp[i][0]
            gas_kinematics_err[i, :, :] = ppxf_tmp[i][1]
            chi2[i] = ppxf_tmp[i][2]
            gas_flux[i, :] = ppxf_tmp[i][3]
            gas_flux_error[i, :] = ppxf_tmp[i][4]
            bestfit[i, :] = ppxf_tmp[i][6]
            gas_bestfit[i, :] = ppxf_tmp[i][7]
            stkin[i, :] = ppxf_tmp[i][8]
            stkin_err[i, :] = ppxf_tmp[i][9]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, np.max(ubins) + 1):
            # start[0]=stellar_kinematics[i, :]
            start[0] = start[i]  # Added this in. Check if ok.

            (
                gas_kinematics[i, :, :],
                gas_kinematics_err[i, :, :],
                chi2[i],
                gas_flux[i, :],
                gas_flux_error[i, :],
                _,
                bestfit[i, :],
                gas_bestfit[i, :],
                stkin[i, :],
                stkin_err[i, :],
            ) = run_ppxf(
                templates,
                spectra[:, i],
                error[:, i],
                velscale,
                start[i],
                goodPixels_gas,
                tpl_comp,
                moments,
                offset,
                emi_mpol_deg,
                fixed[i],
                velscale_ratio,
                tied,
                gas_comp,
                gas_names,
                i,
                nbins,
                ubins,
            )

        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print(
        "             Running PPXF on %s spectra took %.2fs"
        % (nbins, time.time() - start_time)
    )
    # print("")
    logging.info(
        "Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check if there was a problem with a spectra: NOT DONE

    # # add back the part of the spectrum that was truncated because of lack of templates - needed?
    # bestfit_1 = np.zeros((nbins,npix_in))
    # gas_bestfit_1 = np.zeros((nbins,npix_in))
    # bestfit_1[:, wav_cov_templates]=bestfit
    # gas_bestfit_1[:, wav_cov_templates]=gas_bestfit

    # tidy up the ppXF output so it matches the order to the original line-list
    (
        linesfitted,
        fluxes_final,
        fluxes_err_final,
        vel_final,
        vel_err_final,
        sigma_final,
        sigma_err_final,
        extra,
    ) = tidy_up_fluxes_and_kinematics(
        gas_kinematics,
        gas_kinematics_err,
        gas_flux,
        gas_flux_error,
        emldb,
        eml_tying,
        config["GAS"]["MOM"],
    )

    # get fluxes in the correct units, see Westfall et al. 2019 eq 16
    gas_flux_in_units = (
        fluxes_final
        * (velscale / C)
        * linesfitted["lambda"]
        * (1 + config["GENERAL"]["REDSHIFT"] / C)
    )
    gas_err_flux_in_units = (
        fluxes_err_final
        * (velscale / C)
        * linesfitted["lambda"]
        * (1 + config["GENERAL"]["REDSHIFT"] / C)
    )
    # divide by the number of spaxels per bin to make the flux per spaxel
    for i in range(gas_flux_in_units.shape[0]):
        gas_flux_in_units[i, :] = gas_flux_in_units[i, :] / n_spaxels_per_bin[i]
        gas_err_flux_in_units[i, :] = gas_err_flux_in_units[i, :] / n_spaxels_per_bin[i]

    # add back the template LSF
    eml_fwhm_angstr = LSF_Templates(linesfitted["lambda"])
    templates_sigma = eml_fwhm_angstr / linesfitted["lambda"] * C / 2.355

    # templates_sigma = np.zeros(sigma_final.shape)+templates_sigma
    sigma_final_measured = (sigma_final**2 + templates_sigma**2) ** (0.5)

    # save results to file
    if config["GAS"]["LEVEL"] != "BOTH":
        save_ppxf_emlines(
            config,
            config["GENERAL"]["OUTPUT"],
            config["GENERAL"]["RUN_ID"],
            config["GAS"]["LEVEL"],
            linesfitted,
            gas_flux_in_units,
            gas_err_flux_in_units,
            vel_final,
            vel_err_final,
            sigma_final_measured,
            sigma_err_final,
            chi2,
            templates_sigma,
            bestfit,
            gas_bestfit,
            stkin,
            spectra,
            error,
            goodPixels_gas,
            logLam_galaxy,
            ubins,
            npix,
            extra,
        )

    if (
        config["GAS"]["LEVEL"] == "BOTH"
    ):  # Special case when wanting the gas in bin and spaxel modes
        save_ppxf_emlines(
            config,
            config["GENERAL"]["OUTPUT"],
            config["GENERAL"]["RUN_ID"],
            "BIN",
            linesfitted,
            gas_flux_in_units,
            gas_err_flux_in_units,
            vel_final,
            vel_err_final,
            sigma_final_measured,
            sigma_err_final,
            chi2,
            templates_sigma,
            bestfit,
            gas_bestfit,
            stkin,
            spectra,
            error,
            goodPixels_gas,
            logLam_galaxy,
            ubins,
            npix,
            extra,
        )

        save_ppxf_emlines(
            config,
            config["GENERAL"]["OUTPUT"],
            config["GENERAL"]["RUN_ID"],
            "SPAXEL",
            linesfitted,
            gas_flux_in_units,
            gas_err_flux_in_units,
            vel_final,
            vel_err_final,
            sigma_final_measured,
            sigma_err_final,
            chi2,
            templates_sigma,
            bestfit,
            gas_bestfit,
            stkin,
            spectra,
            error,
            goodPixels_gas,
            logLam_galaxy,
            ubins,
            npix,
            extra,
        )

    printStatus.updateDone("Emission line fitting done")
    # print("")
    logging.info("Emission Line Fitting done\n")
