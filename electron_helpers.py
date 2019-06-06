from read_root_output import make_irfs_from_root
from scipy.ndimage import median_filter
from electron_fitter import *
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm as gaus
from numpy.random import poisson

__all__ = ['create_fitter', 'spectral_feature_significance', 'observation_time_cutoff',
           'observation_time_new_power_law', 'make_sensitivity_curve',
           'get_spectral_points', 'get_fit_envelope']


def median_filter_templates(template, filter_size=(2, 3)):
    """

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
    template_filter = median_filter(template_scaled, filter_size) * scale_fac

    return template_filter


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
    electron_template = median_filter_templates(electron_template[1], filter_size=(1, 3))
    proton_template = median_filter_templates(proton_template[1], filter_size=(1, 3))

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

    template_rand = poisson(total_template[0] * time)

    energy, flux, error = spec_fitter.get_spectral_points(template_rand, time,
                                                          electron_spectrum,
                                                          electron_spectrum_parameters,
                                                          proton_spectrum,
                                                          proton_spectrum_parameters)

    return energy, flux, error


def get_fit_envelope(total_template, spec_fitter, time,
                     electron_spectrum, electron_spectrum_parameters,
                     proton_spectrum=None, proton_spectrum_parameters=None,
                     num_eval=100, energies=np.logspace(-2,3,1000)):

    envelope = []
    for i in range(num_eval):
        template_rand = poisson(total_template[0] * time)

        vals, errors, like = spec_fitter.fit_model(template_rand, time,
                                                   electron_spectrum,
                                                   electron_spectrum_parameters,
                                                   proton_spectrum,
                                                   proton_spectrum_parameters)
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

    # First create the
    template, fitter = create_fitter(electron_file, proton_file,
                                     electron_spectrum_simulation,
                                     electron_simulation_parameters,
                                     power_law_proton, {},
                                     telescope_multiplicity=telescope_multiplicity)
    lr = []
    for i in range(iterations):
        template_rand = poisson(template[0] * observation_time)
        vals, errors, likelihood = fitter.fit_model(template_rand, observation_time,
                                                    electron_spectrum_null_model,
                                                    electron_null_parameters,
                                                    energy_range=energy_range)
        vals, errors, likelihood2 = fitter.fit_model(template_rand, observation_time,
                                                     electron_spectrum_test_model,
                                                     electron_test_parameters,
                                                     energy_range=energy_range)

        if np.isnan(likelihood) or np.isnan(likelihood2):
            continue
        lr.append(likelihood - likelihood2)

    med_diff = np.mean(lr)

    return gaus.isf(chi2.pdf(med_diff, degrees_of_freedom))


def observation_time_cutoff(electron_file, proton_file, cutoff, cut_power,
                            sig_threshold=3, sig_accuracy=0.01,
                            min_iterations=100, start_time=100 * 3600):
    """

    :param cutoff:
    :param cut_power:
    :param sig_threshold:
    :param sig_accuracy:
    :param min_iterations:
    :param start_time:
    :return:
    """

    time = start_time

    sig = 0
    time_est = time
    iterations = 20

    while np.abs(sig_threshold - sig) > sig_accuracy or iterations < min_iterations:
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
        sigma_sqrt_time = sig / np.sqrt(time)
        if np.isinf(sig):
            time_est = time / 10
        elif np.isnan(sig):
            time_est = time * 10
        else:
            time_est = np.power(sig_threshold / sigma_sqrt_time, 2)

        print(("Sig:", sig, "Time:", time / 3600., "Iterations:", iterations))
        iterations = int(iterations * 1.5)
    return time


def observation_time_new_power_law(electron_file, proton_file, norm, index,
                                   decorr_energy, sig_threshold=3,
                                   sig_accuracy=0.01, min_iterations=100,
                                   start_time=100 * 3600):
    """

    :param norm:
    :param index:
    :param decorr_energy:
    :param sig_threshold:
    :param sig_accuracy:
    :param min_iterations:
    :param start_time:
    :return:
    """

    time = start_time

    sig = 0
    time_est = time
    iterations = 20

    while np.abs(sig_threshold - sig) > sig_accuracy or iterations < min_iterations:
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

        sigma_sqrt_time = sig / np.sqrt(time)
        if np.isinf(sig):
            time_est = time / 10
        elif np.isnan(sig):
            time_est = time * 10
        else:
            time_est = np.power(sig_threshold / sigma_sqrt_time, 2)

        print(("Sig:", sig, "Time:", time / 3600., "Iterations:", iterations,
              "Est time", time_est / 3600.))
        iterations = int(iterations * 1.3)
    return time


def make_sensitivity_curve(electron_file, proton_file, time,
                           energy_bins=np.logspace(-2, 2, 21),
                           sig_threshold=5, sig_accuracy=0.01, min_iterations=100,
                           output_file=None,
                           telescope_multiplicity=5):

    sensitivity = []
    bin_centre = np.power(10, np.log10(energy_bins[:-1]) +
                          np.diff(np.log10(energy_bins)) / 2.)

    for energy_bin in range(energy_bins.shape[0] - 1):
        iterations = 50
        sig = 0
        flux_est = 1.

        energy_range = (energy_bins[energy_bin], energy_bins[energy_bin + 1])
        energy_centre = np.power(10, np.log10(energy_bins[energy_bin]) +
                                 (np.log10(energy_bins[energy_bin + 1]) - np.log10(
                                     energy_bins[energy_bin])) / 2)
        count = 0
        while np.abs(sig_threshold - sig) > sig_accuracy or iterations < min_iterations:
            flux = flux_est
            if flux < 1e-6 or flux > 1e6 or count>50:
                sensitivity.append(np.nan)
                break

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

            if np.isinf(sig):
                flux_est = flux / 10
            elif np.isnan(sig):
                flux_est = flux * 10
            elif sig == 0.0:
                flux_est = flux * 10
            else:
                flux_est = flux * np.power(sig_threshold / sig, 1)
                iterations = int(iterations * 1.3)
                if iterations > 5000:
                    iterations = 5000

            print(("Sig:", sig, "Flux:", flux, "Iterations:", iterations, "Est flux",
                  flux_est))
            count += 1

        sensitivity.append(flux * hess_electron_spectrum(energy_centre))
        print((energy_range, sig, energy_centre, flux))

    print((bin_centre.shape, np.array(sensitivity).shape))
    if output_file:
        np.savetxt(output_file, np.vstack((bin_centre, np.array(sensitivity))).T)

    return bin_centre, sensitivity