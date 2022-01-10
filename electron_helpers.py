"""
Helper functions for producing the plots required for the paper
"""
from read_root_output import make_irfs_from_root, make_irfs_from_pandas
from scipy.ndimage import median_filter
from electron_fitter import *
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm as gaus
from numpy.random import poisson

__all__ = ['create_fitter', 'spectral_feature_significance', 'observation_time_cutoff',
           'observation_time_new_power_law', 'make_sensitivity_curve',
           'get_spectral_points', 'get_fit_envelope', 'median_filter_templates']


def median_filter_templates(template, filter_size=(2, 3)):
    """
    Perform simple median filtering on the classifier templates

    :param template: ndarray
        2D classifier template
    :param filter_size: (int, int)
        Filter X and Y sizes
    :return: ndarray
        Filtered template
    """
    template_used = template
    scale_fac = np.sum(template_used, axis=1)[..., np.newaxis]
    template_scaled = template_used / scale_fac
    template_filter = median_filter(template_scaled, filter_size)
    template_filter /= np.sum(template_filter, axis=1)[..., np.newaxis]

    return template_filter * scale_fac


def create_fitter_prod3(electron_file, electron_root_file, proton_file, proton_root_file,
                        electron_spectrum, electron_spectrum_parameters,
                        proton_spectrum, proton_spectrum_parameters,
                        telescope_multiplicity=2, angular_cut=4, zeta_range=(0.2, 1),
                        energy_resolution=None):
    """
    Funtion creates a spectrum fitting object, given a simulated spectral shape and
    parameters and input files

    :param electron_file: str
        Electron file name
    :param proton_file: str
        Proton file name
    :param electron_spectrum: function_ptr
        Pointer to electron spectrum function
    :param electron_spectrum_parameters: dict
            Dictionary of electron spectrum function inputs
    :param proton_spectrum: function_ptr
        Pointer to proton spectrum function
    :param proton_spectrum_parameters: dict
        Dictionary of proton spectrum function inputs
    :param telescope_multiplicity: int
        Minimum telescope multiplicity
    :param angular_cut: float
        Maximum reconstructed event offset used
    :param zeta_range: (float, float)
        Range of zeta parameter used in templates
    :return: (templates, ElectronSpectrumFit)
        Templates and electron spectrum fitting object
    """


    # Create IRFs for Electrons and protons with the given spectra
    electron_area, electron_migration, electron_template = make_irfs_from_pandas(
        electron_file, electron_root_file,
        electron_spectrum, electron_spectrum_parameters,
        telescope_multiplicity=telescope_multiplicity,
        angular_cut=angular_cut, zeta_range=zeta_range,
        energy_resolution=energy_resolution)

    proton_area, proton_migration, proton_template = make_irfs_from_pandas(
        proton_file, proton_root_file, proton_spectrum,
        proton_spectrum_parameters,
        telescope_multiplicity=telescope_multiplicity,
        angular_cut=angular_cut,
        zeta_range=zeta_range)

    # Aplly our median filter to reduce fluctutations
    electron_template = median_filter_templates(electron_template[1], filter_size=(1, 3))
    proton_template = median_filter_templates(proton_template[1], filter_size=(1, 1))
    #electron_template = electron_template[1]
    #proton_template = proton_template[1]

    # Create spectral fitter object
    spec_fitter = ElectronSpectrumFit(electron_migration[1], electron_area[1],
                                      electron_template, proton_template,
                                      electron_migration[0][0], electron_migration[0][1])

    # Sum our templates to make an total measured expectation
    total_template = electron_template + proton_template

    # Return templates and fitting object
    return (total_template, electron_template, proton_template), spec_fitter


def create_fitter(electron_file, proton_file,
                  electron_spectrum, electron_spectrum_parameters,
                  proton_spectrum, proton_spectrum_parameters,
                  telescope_multiplicity=2, angular_cut=4, zeta_range=(0.2, 1),
                  energy_resolution=None):
    """
    Funtion creates a spectrum fitting object, given a simulated spectral shape and
    parameters and input files

    :param electron_file: str
        Electron file name
    :param proton_file: str
        Proton file name
    :param electron_spectrum: function_ptr
        Pointer to electron spectrum function
    :param electron_spectrum_parameters: dict
            Dictionary of electron spectrum function inputs
    :param proton_spectrum: function_ptr
        Pointer to proton spectrum function
    :param proton_spectrum_parameters: dict
        Dictionary of proton spectrum function inputs
    :param telescope_multiplicity: int
        Minimum telescope multiplicity
    :param angular_cut: float
        Maximum reconstructed event offset used
    :param zeta_range: (float, float)
        Range of zeta parameter used in templates
    :return: (templates, ElectronSpectrumFit)
        Templates and electron spectrum fitting object
    """


    # Create IRFs for Electrons and protons with the given spectra
    electron_area, electron_migration, electron_template = make_irfs_from_root(
        electron_file,
        electron_spectrum, electron_spectrum_parameters,
        telescope_multiplicity=telescope_multiplicity,
        angular_cut=angular_cut, zeta_range=zeta_range,
        energy_resolution=energy_resolution)

    proton_area, proton_migration, proton_template = make_irfs_from_root(
        proton_file, proton_spectrum,
        proton_spectrum_parameters,
        telescope_multiplicity=telescope_multiplicity,
        angular_cut=angular_cut,
        zeta_range=zeta_range)

    # Aplly our median filter to reduce fluctutations
    electron_template = median_filter_templates(electron_template[1], filter_size=(3, 1))
    proton_template = median_filter_templates(proton_template[1], filter_size=(3, 1))

    # Create spectral fitter object
    spec_fitter = ElectronSpectrumFit(electron_migration[1], electron_area[1],
                                      electron_template, proton_template,
                                      electron_migration[0][0], electron_migration[0][1])

    # Sum our templates to make an total measured expectation
    total_template = electron_template + proton_template

    # Return templates and fitting object
    return (total_template, electron_template, proton_template), spec_fitter


def get_spectral_points(total_template, spec_fitter, time,
                           electron_spectrum, electron_spectrum_parameters,
                           proton_spectrum=None, proton_spectrum_parameters=None):
    """
    Create realisation of a spectral points for a given spectral model

    :param total_template: ndarray
        2D template for fitting
    :param spec_fitter: ElectronFitter
        ElectronFitter object to perform model fit
    :param time: float
        Simulated time (seconds)
    :param electron_spectrum: function_ptr
        Pointer to electron spectrum function
    :param electron_spectrum_parameters: dict
        Dictionary of electron spectrum function inputs
    :param proton_spectrum: function_ptr
        Pointer to proton spectrum function
    :param proton_spectrum_parameters: dict
        Dictionary of proton spectrum function inputs
    :return:
        (energy bin centre, flux points, flux error)
    """

    # Create Poissonian realisation of the measured template
    input_template = total_template[0].astype(np.float64) * time
    input_template[np.isnan(input_template)] = 0
    template_rand = poisson(input_template.astype(np.float64))

    # Perform spectral fit
    vals, errors, like = spec_fitter.fit_model(template_rand, time,
                                               electron_spectrum,
                                               electron_spectrum_parameters,
                                               proton_spectrum,
                                               proton_spectrum_parameters)

    energy, flux, error = spec_fitter.get_spectral_points(template_rand, time,
                                                          electron_spectrum,
                                                          vals,
                                                          proton_spectrum,
                                                          proton_spectrum_parameters)

    return energy, flux, error


def get_fit_envelope(total_template, spec_fitter, time,
                     electron_spectrum, electron_spectrum_parameters,
                     proton_spectrum=None, proton_spectrum_parameters=None,
                     num_eval=100, energies=np.logspace(-2,3,1000)):
    """
    Create envelope of a number of spectral sits to different realisations of the
    templates

    :param total_template: ndarray
        2D template for fitting
    :param spec_fitter: ElectronFitter
        ElectronFitter object to perform model fit
    :param time: float
        Simulated time (seconds)
    :param electron_spectrum: function_ptr
        Pointer to electron spectrum function
    :param electron_spectrum_parameters: dict
        Dictionary of electron spectrum function inputs
    :param proton_spectrum: function_ptr
        Pointer to proton spectrum function
    :param proton_spectrum_parameters: dict
        Dictionary of proton spectrum function inputs
    :param num_eval: int
        Number of realisations to fit
    :param energies: ndarray
        Energies at which to evaluate model (TeV)
    :return:
        (energy, fit envelope)
    """

    envelope = []

    # Loop over our number of required iterations
    for i in range(num_eval):
        # Create a random realisation
        input_template = total_template[0].astype(np.float64) * time
        input_template[np.isnan(input_template)] = 0

        template_rand = poisson(input_template.astype(np.float64) )
        # Fit the model to this
        vals, errors, like = spec_fitter.fit_model(template_rand, time,
                                                   electron_spectrum,
                                                   electron_spectrum_parameters,
                                                   proton_spectrum,
                                                   proton_spectrum_parameters)

        # Evaluate the model at our energies and add to the envelope
        envelope.append(electron_spectrum(energies, **vals))

    return energies, np.array(envelope)


def spectral_feature_significance(electron_file, proton_file,
                                  observation_time,
                                  electron_spectrum_simulation,
                                  electron_simulation_parameters,
                                  electron_spectrum_null_model, electron_null_parameters,
                                  electron_spectrum_test_model, electron_test_parameters,
                                  proton_spectrum_model=None, proton_parameters=None,
                                  iterations=10, degrees_of_freedom=2,
                                  energy_range=None, telescope_multiplicity=5):
    """
    Calculate the improvement in significance between two different spectral fits

    :param electron_file: str
        Electron file name
    :param proton_file: str
        Proton file name
    :param observation_time: float
        Observation time (seconds)
    :param electron_spectrum_simulation: function_ptr
        Spectral model to simulate
    :param electron_simulation_parameters: dict
        Spectral parameters to simulate
    :param electron_spectrum_null_model: function_ptr
        Spectral model of null model tested
    :param electron_null_parameters: dict
        Spectral parameters of null model tested
    :param electron_spectrum_test_model: function_ptr
        Spectral model of second model tested
    :param electron_test_parameters: dict
        Spectral parameters of second model tested
    :param proton_spectrum_model: function_ptr
        Fitted proton spectrum
    :param proton_parameters: dict
        Proton spectral parameters
    :param iterations: int
        Number of iterations to evaluate significance
    :param degrees_of_freedom: int
        Degrees of freedom between two model fits
    :param energy_range: tuple
        Lower and upper energy range of fit (TeV)
    :param telescope_multiplicity: int
        Minimum telescope multiplicity required
    :return: float
        Significance
    """

    # First create the fitter object
    template, fitter = create_fitter(electron_file, proton_file,
                                     electron_spectrum_simulation,
                                     electron_simulation_parameters,
                                     power_law_proton, {},
                                     telescope_multiplicity=telescope_multiplicity)
    lr = [] # Likelihood ration list

    # Loop over required iterations
    for i in range(iterations):

        # Create a random realisation
        #template_rand = poisson(template[0].astype(np.float64) * observation_time)
        input_template = template[0].astype(np.float64) * observation_time
        input_template[np.isnan(input_template)] = 0

        template_rand = poisson(input_template.astype(np.float64) )
        # Fit our null model to this realisation
        vals, errors, likelihood = fitter.fit_model(template_rand, observation_time,
                                                    electron_spectrum_null_model,
                                                    electron_null_parameters,
                                                    energy_range=energy_range)
        # Then our test model
        vals, errors, likelihood2 = fitter.fit_model(template_rand, observation_time,
                                                     electron_spectrum_test_model,
                                                     electron_test_parameters,
                                                     energy_range=energy_range)

        # If the fit didn't work ignore it
        if np.isnan(likelihood) or np.isnan(likelihood2):
            continue
        # Create likelihood ratio
        lr.append(likelihood - likelihood2)

    # Take the mean of this value
    med_diff = np.mean(lr)

    # Return the gaussian significance
    return gaus.isf(chi2.pdf(med_diff, degrees_of_freedom))


def estimate_time_requirement(time, significance, target_significance=5):
    """
    Estimate the required time needed to reach a given significance level,
    given current significance

    :param time: float
        Time of current simulation
    :param significance: float
        Significance of current simulation
    :param target_significance: float
        Target significance level to reach
    :return: float
        Time estimate
    """

    sigma_sqrt_time = significance / np.sqrt(time)

    # If significance is infinite likely our time was too long
    if np.isinf(significance):
        time_est = time / 10
    # And if NaN too short
    elif np.isnan(significance):
        time_est = time * 10
    else:
        # Otherwise time estimate scale by square of significance
        time_est = np.power(target_significance / sigma_sqrt_time, 2)

    return time_est


def estimate_flux_requirement(flux, significance, target_significance=5):
    """
    Estimate the required source flux needed to reach a given significance level,
    given current significance

    :param flux: float
        Flux level of current simulation (arb units)
    :param significance: float
        Significance of current simulation
    :param target_significance:  float
        Target significance level to reach
    :return: float
        Flux estimate
    """

    # If significance is infinite likely our time was too long
    if np.isinf(significance):
        flux_est = flux / 10
    # And if NaN too short
    elif np.isnan(significance):
        flux_est = flux * 10
    # If 0 also too short
    elif significance == 0.0:
        flux_est = flux * 10
    else:
        # Otherwise significance scales linearly with flux
        flux_est = flux * np.power(target_significance / significance, 1)

    return flux_est


def observation_time_cutoff(electron_file, proton_file, cutoff, cut_power,
                            sig_threshold=3, sig_accuracy=0.01,
                            min_iterations=100, start_time=100 * 3600,
                            telescope_multiplicity=5):
    """
    Calculate time requirement to observe a cutoff on top of a HESS-like spectrum at a
    given significance level

    :param electron_file: str
        Electron input file name
    :param proton_file: str
        Proton input file name
    :param cutoff: float
        Cut off energy (TeV)
    :param cut_power: float
        Cut off power (TeV)
    :param sig_threshold: float
        Significance thrshold required
    :param sig_accuracy: float
        Accuracy of significance value
    :param min_iterations: int
        Minimum number of iterations required before value returned
    :param start_time: float
        Starting time tested
    :param telescope_multiplicity: int
        Minimum telescope multiplicity required
    :return: float
        Time to reach significance
    """

    time = start_time

    sig = 0
    time_est = time
    iterations = 20

    while np.abs(sig_threshold - sig) > sig_accuracy or iterations < min_iterations:
        iterations = int(iterations * 1.5)

        time = time_est
        sig = spectral_feature_significance(electron_file, proton_file, time,
                                            electrons_plus_cut_off,
                                            {"electron_index2": 3.78,
                                             "inverse_cut": cutoff,
                                             "cut_power": cut_power},
                                            hess_electron_spectrum,
                                            {"electron_index2": (3.78, (3., 5))},
                                            electrons_plus_cut_off,
                                            {"electron_index2": (3.78, (3., 5.)),
                                             "inverse_cut": (cutoff, (0., 1)),
                                             "cut_power": (3, (0.5, 5))},
                                            iterations=iterations)

        time_est = estimate_time_requirement(time, sig, sig_threshold)

        print(("Sig:", sig, "Time:", time / 3600., "Iterations:", iterations))
    return time


def observation_time_new_power_law(electron_file, proton_file, norm, index,
                                   decorr_energy, sig_threshold=3,
                                   sig_accuracy=0.01, min_iterations=100,
                                   start_time=100 * 3600, telescope_multiplicity=5):
    """
    Calculate time requirement to observe a new power-law on top of a HESS-like spectrum
    at a given significance level

    :param electron_file: str
        Electron input file name
    :param proton_file: str
        Proton input file name
    :param norm: float
        Power law normalisation
    :param index: float
        Power law index
    :param decorr_energy: float
        Decorreleation energy of the power law
    :param sig_threshold: float
        Significance thrshold required
    :param sig_accuracy: float
        Accuracy of significance value
    :param min_iterations: int
        Minimum number of iterations required before value returned
    :param start_time: float
        Starting time tested
    :param telescope_multiplicity: int
        Minimum telescope multiplicity required
    :return: float
        Time to reach significance
    """

    time = start_time

    sig = 0
    time_est = time
    iterations = 20

    while np.abs(sig_threshold - sig) > sig_accuracy or iterations < min_iterations:
        iterations = int(iterations * 1.5)

        time = time_est
        sig = spectral_feature_significance(electron_file, proton_file, time,
                                            electrons_plus_power_law,
                                            {"electron_index2": 3.78,
                                             "power_law_alpha": index,
                                             "power_law_normalisation": norm,
                                             "power_law_decorrelation_energy":
                                                 decorr_energy},
                                            hess_electron_spectrum,
                                            {"electron_index2": (3.78, (1., 5))},
                                            electrons_plus_power_law,
                                            {"electron_index2": (3.78, (1., 5.)),
                                             "power_law_alpha": (index, (0, 3.5)),
                                             "power_law_normalisation": (norm, (0, 1e-8)),
                                             "power_law_decorrelation_energy": (
                                             decorr_energy,
                                             (decorr_energy, decorr_energy))},
                                            iterations=iterations)

        time_est = estimate_time_requirement(time, sig, sig_threshold)

        print(("Sig:", sig, "Time:", time / 3600., "Iterations:", iterations,
              "Est time", time_est / 3600.))
    return time


def make_sensitivity_curve(electron_file, proton_file, time,
                           energy_bins=np.logspace(-2, 2, 21),
                           sig_threshold=5, sig_accuracy=0.01, min_iterations=100,
                           output_file=None,
                           telescope_multiplicity=5):
    """
    Create electron flux sensitivity curve, i.e. the flux required to reach a given
    significance level within a time period

    :param electron_file: str
        Electron input file name
    :param proton_file: str
        Proton input file name
    :param time: float
        Time at which to evaluate sensitivity
    :param energy_bins: ndarray
        Energy bins in which to calculate sensitivity
    :param sig_threshold: float
        Significance thrshold required
    :param sig_accuracy: float
        Accuracy of significance value
    :param min_iterations: int
        Minimum number of iterations required before value returned
    :param output_file:
        Output file name
    :param telescope_multiplicity: int
        Minimum telescope multiplicity required
    :return: (ndarray, ndarray)
        (Centres of energy bins, sensitivity values)
    """

    sensitivity = []
    # First calculate our energy bin centres
    bin_centre = np.power(10, np.log10(energy_bins[:-1]) +
                          np.diff(np.log10(energy_bins)) / 2.)

    # Now loop over our energy bins
    for energy_bin in range(energy_bins.shape[0] - 1):
        iterations = 50
        sig = 0
        flux_est = 1.

        energy_range = (energy_bins[energy_bin], energy_bins[energy_bin + 1])
        energy_centre = np.power(10, np.log10(energy_bins[energy_bin]) +
                                 (np.log10(energy_bins[energy_bin + 1]) - np.log10(
                                     energy_bins[energy_bin])) / 2)

        # Now loop in each bin to calculate the flux required to reach significance level
        count = 0
        while np.abs(sig_threshold - sig) > sig_accuracy or iterations < min_iterations:
            flux = flux_est
            if flux < 1e-6 or flux > 1e6 or count > 10:
                sensitivity.append(np.nan)
                break

            # Calculate the significance of the electron spectrum in this energy bin in
            # comparison to no electrons
            sig = spectral_feature_significance(electron_file, proton_file, time,
                                                hess_electron_spectrum,
                                                {
                                                "electron_normalisation": flux * 1.049e-4},
                                                hess_electron_spectrum,
                                                {"electron_normalisation": (
                                                0., (0., 0.))},
                                                hess_electron_spectrum,
                                                {"electron_normalisation": (
                                                flux * 1.049e-4, (flux * 1.049e-4,
                                                                  flux * 1.049e-4))},
                                                iterations=iterations,
                                                energy_range=energy_range,
                                                telescope_multiplicity=
                                                telescope_multiplicity)


            # Estimate the next flux to test
            flux_est = estimate_flux_requirement(flux, sig, sig_threshold)

            if np.isnan(sig) or np.isinf(sig):
                count += 1
            else:
                count = 0
                iterations = int(iterations * 1.3)

            if iterations > 5000:
                iterations = 5000

            print(("Sig:", sig, "Flux:", flux, "Iterations:", iterations, "Est flux",
                  flux_est))

        sensitivity.append(flux * hess_electron_spectrum(energy_centre))

    # Save the output file if we want one
    if output_file:
        np.savetxt(output_file, np.vstack((bin_centre, np.array(sensitivity))).T)

    return bin_centre, sensitivity
