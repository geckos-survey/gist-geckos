import glob
import os
import numpy as np
from astropy.io import fits
import logging

from printStatus import printStatus

from ppxf.ppxf_util import log_rebin, gaussian_filter1d

#BPASS age range, nothing older than 12.59Gyr
logAgesAll = np.arange(6,11+0.1,0.1)
agesFlags = (logAgesAll<=10.2)


### USED TO SET THE DESIRED AGES, METALLICITIES, AND ALPHAS
#Default values
listLogAge = logAgesAll[0::2]										#0.2dex age steps 1.25 Myr -> 15.84 Gyr
listMetal = np.array([1.e-4,1.e-3,2.e-3,4.e-3,8.e-3,1.4e-2,2.e-2,4.e-2])
listAlpha = np.array([0.])

#Needed by both functions to select ages
logAgeSteps = np.sum([np.isclose(logAgesAll,aa) for aa in listLogAge],axis=0,dtype=bool)* agesFlags 


def age_metal_alpha(passedFiles):
	"""
	Function to extract the values of age, metallicity, and alpha-enhancement
	from standard BPASS filenames. Note that this function can automatically
	distinguish between template libraries that do or do not include
	alpha-enhancement.
	"""

	files = []
	for i in range( len(passedFiles) ):
		files.append( passedFiles[i].split('/')[-1] )

	Metal = np.zeros(len(files))
	Alpha = np.zeros(len(files))
	metal_str = np.array([],dtype='str')
	alpha_str = np.array([],dtype='str')
	for ff, file in enumerate(files):
		
		a = file.find(".a")
		alpha_str = np.append(alpha_str,file[a+1 : a+5])
		Alpha[ff] = float(file[a+2 : a+5])

		z = file.find('.z')
		metal_str = np.append(metal_str,file[z+1 : z+5])
		z = file[z+2 : z+5]
		if 'em' in z:
			z = z[-1]
			Metal[ff] = 1.*10**(-1*float(z))
		else:
			Metal[ff] = 1.e-3 * float(z)
	
	logAge = logAgesAll[logAgeSteps] -9

	metalSteps = np.sum([np.isclose(Metal,zz) for zz in listMetal],axis=0,dtype=bool)
	Metal, indMetal = np.unique(Metal[metalSteps],return_index=True)
	metal_str = metal_str[metalSteps][indMetal]

	alphaSteps = np.sum([np.isclose(Alpha,zz) for zz in listAlpha],axis=0,dtype=bool)
	Alpha, indAlpha = np.unique(Alpha[alphaSteps],return_index=True)
	alpha_str = alpha_str[alphaSteps][indAlpha]


	nAges = len(logAge)
	nMetal = len(Metal)
	nAlpha = len(Alpha)
	ncomb = nAges*nMetal*nAlpha



	return(logAge, Metal, Alpha, metal_str, alpha_str, nAges, nMetal, nAlpha, ncomb)



def prepareSpectralTemplateLibrary(config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid):
	"""
	Prepares the spectral template library. The templates are loaded from disk,
	shortened to meet the spectral range in consideration, convolved to meet the
	resolution of the observed spectra (according to the LSF), log-rebinned, and
	normalised. In addition, they are sorted in a three-dimensional array
	sampling the parameter space in age, metallicity and alpha-enhancement.
	"""
	printStatus.running("Preparing the stellar population templates")
	cvel  = 299792.458

	# SSP model library
	sp_models = glob.glob(os.path.join(config['GENERAL']['TEMPLATE_DIR'],config[module_used]['LIBRARY'])+'*.dat')


	sp_models.sort()

	# Read data
	ssp 		   = np.loadtxt(sp_models[0])
	ssp_data       = ssp[:,1]
	lamRange_spmod = ssp[[0,-1],0]


	# Determine length of templates
	template_overhead = np.zeros(2)
	if lmin - lamRange_spmod[0] > 150.:
	    template_overhead[0] = 150.
	else:
	    template_overhead[0] = lmin - lamRange_spmod[0] - 5
	if lamRange_spmod[1] - lmax > 150.:
	    template_overhead[1] = 150.
	else:
	    template_overhead[1] = lamRange_spmod[1] - lmax - 5

	# Shorten templates to size of data
	# Reconstruct full original lamRange
	ssp_waveStep = np.diff(ssp[:,0])[0]
	lamRange_lin = np.arange( lamRange_spmod[0], lamRange_spmod[-1]+ssp_waveStep, ssp_waveStep )
	# Create new lamRange according to the provided LMIN and LMAX values, according to the module which calls
	constr = np.array([ lmin - template_overhead[0], lmax + template_overhead[1] ])
	idx_lam = np.where( np.logical_and(lamRange_lin > constr[0], lamRange_lin < constr[1] ) )[0]
	lamRange_spmod = np.array([ lamRange_lin[idx_lam[0]], lamRange_lin[idx_lam[-1]] ])
	# Shorten data to size of new lamRange
	ssp_data = ssp_data[idx_lam]

	# Convolve templates to same resolution as data
	if len( np.where( LSF_Data(lamRange_lin[idx_lam]) - LSF_Templates(lamRange_lin[idx_lam]) < 0. )[0] ) != 0:
		message = "According to the specified LSF's, the resolution of the "+\
					"templates is lower than the resolution of the data. Exit!"
		printStatus.updateFailed("Preparing the stellar population templates")
		print("             "+message)
		logging.critical(message)
		exit(1)
	else:
		FWHM_dif = np.sqrt( LSF_Data(lamRange_lin[idx_lam])**2 - LSF_Templates(lamRange_lin[idx_lam])**2 )
		sigma = FWHM_dif/2.355/ssp_waveStep

	# Create an array to store the templates
	sspNew, _, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale)

	# Extract ages, metallicities and alpha from the templates
	# ABW done in all to allow for BPASS Z,age,alpha selection
	logAge, metal, alpha, metal_str, alpha_str, nAges, nMetal, nAlpha, ncomb = age_metal_alpha(sp_models)
	
	# exit()

	ntemplates = nAges*len(sp_models)

	# Do NOT sort the templates in any way
	if sortInGrid == False:

		# Load templates, convolve and log-rebin them
		templates = np.empty((sspNew.size, ntemplates))
		# Sort the templates in the cube of age, metal, alpha
		# This sorts for alpha
		for i, a in enumerate(alpha_str):
			# This sorts for metals
			for k, mh in enumerate(metal_str):
				files = [s for s in sp_models if (mh in s and a in s)]
				# This sorts for ages
				for j, filename in enumerate(files):
					ssp_data = (np.loadtxt(filename)[idx_lam,1:])[:,logAgeSteps]
					for s in range(ssp_data.shape[1]):
						# ssp_data[:,s] = gaussian_filter1d(ssp_data[:,s], sigma)
						# sspNew, logLam2, _ = log_rebin(lamRange_spmod, ssp_data[:,s], velscale=velscale)
						templates[:, nAges*j + s], logLam_spmod, _ = log_rebin(lamRange_spmod, ssp_data[:,s], velscale=velscale)

		# Normalise templates in such a way to get mass-weighted results
		if config[module_used]['NORM_TEMP'] == 'MASS':
			templates = templates / np.mean( templates )

		# Normalise templates in such a way to get light-weighted results
		if config[module_used]['NORM_TEMP'] == 'LIGHT':
			for i in range( templates.shape[1] ):
				templates[:,i] = templates[:,i] / np.mean(templates[:,i], axis=0)

		printStatus.updateDone("Preparing the stellar population templates")
		logging.info("Prepared the stellar population templates")

		return( templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam_spmod, ntemplates, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan )


	# Sort the templates in a cube of age, metal, alpha for the SFH module
	elif sortInGrid == True:

		templates          = np.zeros((sspNew.size, nAges, nMetal, nAlpha))
		templates[:,:,:,:] = np.nan

		# Arrays to store properties of the models
		logAge_grid = np.empty((nAges, nMetal, nAlpha))
		metal_grid  = np.empty((nAges, nMetal, nAlpha))
		alpha_grid  = np.empty((nAges, nMetal, nAlpha))


		# Sort the templates in the cube of age, metal, alpha
		# This sorts for alpha
		for i, a in enumerate(alpha_str):
			# This sorts for metals
			for k, mh in enumerate(metal_str):
				files = [s for s in sp_models if (mh in s and a in s)]
				# This sorts for ages
				for j, filename in enumerate(files):
					ssp_data = (np.loadtxt(filename)[idx_lam,1:])[:,logAgeSteps]
					for s in range(ssp_data.shape[1]):
						ssp_data[:,s] = gaussian_filter1d(ssp_data[:,s], sigma)
						sspNew, logLam2, _ = log_rebin(lamRange_spmod, ssp_data[:,s], velscale=velscale)

						logAge_grid[s, k, i] = logAge[s]
						metal_grid[s, k, i]  = metal[k]
						alpha_grid[s, k, i]  = alpha[i]

						# Normalise templates for light-weighted results
						if config[module_used]['NORM_TEMP'] == 'LIGHT':
							templates[:, nAges*j+s, k, i] = sspNew / np.mean(sspNew)
						else:
							templates[:, nAges*j+s, k, i] = sspNew

		# Normalise templates for mass-weighted results
		if config[module_used]['NORM_TEMP'] == 'MASS':
			templates = templates / np.mean( templates )

		printStatus.updateDone("Preparing the stellar population templates")
		logging.info("Prepared the stellar population templates")

		return(templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam2, ntemplates, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha)


	# agesFlags = (agesAll<=10.2)


	# wave_step = 1

	# if isinstance(mode,type(None)):
	# 	mode = 'short'		

	# if mode == 'emshort':
	# 	agesList = np.array([7.4,7.6,7.8,8,9.2,9.5,9.7,9.9,10.2])
	# 	timeSteps = np.sum([np.isclose(agesAll,aa) for aa in agesList],axis=0,dtype=bool) * agesFlags 
	# 	zList = np.array([1.e-3,6.e-3,1.4e-2,4.e-2])
	# 	refs = np.sum([np.isclose(Zs,zz) for zz in zList],axis=0,dtype=bool)
	# 	files = files[refs]
	# 	Zs = Zs[refs]
	# 	alphas = alphas[refs]
	# 	ages = agesAll[timeSteps]

	# if mode == 'short':
	# 	timeSteps = ~np.array((np.arange(len(agesAll)))%2,dtype=bool) * agesFlags		#0.2dex age steps
	# 	zList = np.array([1.e-3,6.e-3,1.4e-2,4.e-2])
	# 	refs = np.sum([np.isclose(Zs,zz) for zz in zList],axis=0,dtype=bool)
	# 	files = files[refs]
	# 	Zs = Zs[refs]
	# 	alphas = alphas[refs]
	# 	ages = agesAll[timeSteps]

	# if mode == 'pops':
	# 	timeSteps = ~np.array((np.arange(len(agesAll)))%2,dtype=bool)					#0.2dex age steps
	# 	zList = np.array([1.e-4,1.e-3,2.e-3,4.e-3,8.e-3,1.4e-2,2.e-2,4.e-2])
	# 	refs = np.sum([np.isclose(Zs,zz) for zz in zList],axis=0,dtype=bool)
	# 	files = files[refs]
	# 	Zs = Zs[refs]
	# 	alphas = alphas[refs]
	# 	ages = agesAll[timeSteps]

	# elif mode == 'full':
	# 	timeSteps = np.array(np.ones(len(ages)),dtype=bool) * agesFlags					#0.1dex age steps

	# timeSteps = np.append(np.array([False]),timeSteps)

	# Nspec = len(ages)

	# for ff in range(len(files)):

	# 	file = files[ff]
	# 	data = np.loadtxt(file)

	# 	if ff == 0:
	# 		linLambda_templates = data[:,0]

	# 		wave_shorten = np.logical_and((linLambda_templates>=4000), (linLambda_templates<10000))

	# 		linLambda_templates_trunc = linLambda_templates[wave_shorten]
	# 		templates = np.zeros([linLambda_templates_trunc.shape[0],Nspec*len(files)])

	# 	templates[:,ff*Nspec:(ff+1)*Nspec] = data[wave_shorten,:][:,timeSteps]



	# return  templates, linLambda_templates_trunc,  wave_step