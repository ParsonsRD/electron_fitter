"""

"""
import numpy as np
from scipy.stats import poisson, norm
from iminuit import minimize

__all__ = ['ElectronSpectrumFit', 'hess_electron_spectrum', 'power_law_proton',
           'power_law', 'electrons_plus_power_law', 'electrons_plus_cut_off']


def hess_electron_spectrum(energy, electron_normalisation=1.049e-4,
                           electron_decorrelation_energy = 1,
                           electron_index1=3.04,
                           electron_index2=3.78,
                           electron_break_energy=0.94,
                           electron_break_sharpness=0.12, **kwargs):
    """
    Creates a HESS 2017 (preliminary) type broken power-law electron
    spectrum

    :param energy: ndarray
        Energies at which to evaluate spectrum (TeV)
    :param electron_normalisation: float
        Normalisation of spectrum at de-correlation energy (TeV^-1 cm^-2 s^-1)
    :param electron_decorrelation_energy: float
        De-correlation energy (TeV)
    :param electron_index1: float
        Index of low energy power law
    :param electron_index2: float
        Index of high energy power law
    :param electron_break_energy: float
        Energy of break between the indices (TeV)
    :param electron_break_sharpness: float
        Sharpness of energy break
    :return: ndarray
        Flux level at specified energies
    """

    # Provide HESS electron spectrum in units GeV^-1 cm^-2 s^-1
    return electron_normalisation * \
           np.power(energy/electron_decorrelation_energy, -electron_index1) * \
           np.power(1 + np.power(energy / electron_break_energy,
                                 1 / electron_break_sharpness),
                    -1 * (electron_index2 - electron_index1) * electron_break_sharpness)


def power_law(energy, power_law_alpha=2, power_law_normalisation=1e-10,
              power_law_decorrelation_energy=1, **kwargs):
    """
    Creates a power-law type spectrum

    :param energy: ndarray
        Energies at which to evaluate spectrum (TeV)
    :param power_law_alpha: float
        Index of new power-law component
    :param power_law_normalisation: float
        Normalisation of spectrum at de-correlation energy (TeV^-1 cm^-2 s^-1)
    :param power_law_decorrelation_energy: float
        De-correlation energy (TeV)
    :return: ndarray
        Flux level at specified energies
    """

    # Add a power-law component to the HESS function, its index and flux at 10 TeV
    return power_law_normalisation * \
           np.power(energy / power_law_decorrelation_energy, -power_law_alpha)


def power_law_proton(energy, proton_power_law_alpha=2.7,
                     proton_power_law_normalisation=0.096,
                     proton_power_law_decorrelation_energy=1, **kwargs):
    """
    Creates a power-law type spectrum

    :param energy: ndarray
        Energies at which to evaluate spectrum (TeV)
    :param proton_power_law_alpha: float
        Index of new power-law component
    :param proton_power_law_normalisation: float
        Normalisation of spectrum at de-correlation energy (TeV^-1 cm^-2 s^-1)
    :param proton_power_law_decorrelation_energy: float
        De-correlation energy (TeV)
    :return: ndarray
        Flux level at specified energies
    """

    # Add a power-law component to the HESS function, its index and flux at 10 TeV
    return proton_power_law_normalisation * \
           np.power(energy / proton_power_law_decorrelation_energy,
                    -proton_power_law_alpha)

def electrons_plus_power_law(energy, power_law_alpha=2, power_law_normalisation=1e-10,
                             power_law_decorrelation_energy=10., **kwargs):
    """
    Creates an spectrum on HESS broken power-law type with an additional power-law
    component

    :param energy: ndarray
        Energies at which to evaluate spectrum (TeV)
    :param power_law_alpha: float
        Index of new power-law component
    :param power_law_normalisation: float
        Normalisation of spectrum at de-correlation energy (TeV^-1 cm^-2 s^-1)
    :param power_law_decorrelation_energy: float
        De-correlation energy (TeV)
    :return: ndarray
        Flux level at specified energies
    """
    # Add a power-law component to the HESS function, its index and flux at 10 TeV
    return hess_electron_spectrum(energy, **kwargs) + \
           power_law(energy, power_law_alpha, power_law_normalisation,
                     power_law_decorrelation_energy)


def electrons_plus_cut_off(energy, inverse_cut=1./20, cut_power=1, **kwargs):
    """
    Creates an spectrum on HESS broken power-law type with an additional power-law
    component

    :param energy: ndarray
        Energies at which to evaluate spectrum (TeV)
    :param cut_energy: float
        Energy if exponential cut off (TeV)
    :param cut_power: float
        Power of exponential cut off
    :return: ndarray
        Flux level at specified energies
    """

    # Add a power-law component to the HESS function, its index and flux at 10 TeV
    return hess_electron_spectrum(energy, **kwargs) * \
           np.exp(-1 * np.power(energy * inverse_cut, cut_power))


class ElectronSpectrumFit:
    """
    This class performs the forward folding and spectral fitting of the classifier
    templates in simulated electron spectrum measurements in a similar way to the 2008
    and 2017 HESS electron spectrum measurements.

    This is performed by using the electron IRFs predict the number of electron
    counts in a given energy bins and using this to normalise the electron classifier
    templates. Proton contamination is estimated by either assuming proton events make
    up the remainder of events in the energy bin, or using proton IRFs to fit the
    proton spectrum in parallel.
    """
    def __init__(self, electron_migration_matrix, electron_effective_area,
                 electron_template, proton_template,
                 true_energy_bins, reco_energy_bins,
                 proton_migration_matrix=None, proton_effective_area=None,
                 simulated_cone=10):
        """
        :param electron_migration_matrix: ndarray
            Electron energy migration matrix (true energy, reconstructed energy)
        :param electron_effective_area: ndarray
            Electron effective area (true energy)
        :param electron_template: ndarray
            Electron classifier template (classifier, reconstructed energy)
        :param proton_template:
            Proton classifier template (classifier, reconstructed energy)
        :param true_energy_bins: ndarray
            Bin centres of true energy bins (log10(TeV))
        :param reco_energy_bins: ndarray
            Bin centres of reconstructed energy bins (log10(TeV))
        :param proton_migration_matrix: ndarray
            Proton energy migration matrix (true energy, reconstructed energy)
        :param proton_effective_area: ndarray
            Proton effective area (true energy)
        :param simulated_cone: float
            Opening angle of simulations
        """

        # Copy in electron IRFs
        self.electron_migration_matrix, self.electron_effective_area = \
            electron_migration_matrix, electron_effective_area
        # Copy in proton IRFs (if we want to use them)
        self.proton_migration_matrix, self.proton_effective_area = \
            proton_migration_matrix, proton_effective_area

        # Renormalise the proton and electron templates such that their integral is 1
        # in each energy bin
        self.electron_template = electron_template/\
                                 np.sum(electron_template, axis=1)[..., np.newaxis]
        self.electron_template[np.isnan(self.electron_template)] = 0
        self.proton_template = proton_template/\
                                 np.sum(proton_template, axis=1)[..., np.newaxis]
        self.proton_template[np.isnan(self.proton_template)] = 0

        # Convert the simulated view cone into an angular area
        self.simulated_angular_area = 2*np.pi*(1-np.cos(np.deg2rad(simulated_cone)))
        # Copy in our energy binning in reconstructed and true axes
        self.true_energy_bins, self.reco_energy_bins = np.array(true_energy_bins), \
                                                       np.array(reco_energy_bins)

        self.total_distribution = None
        self.time = None
        self.electron_fit_function, self.proton_fit_function = None, None
        self.variable_names, self.energy_range  = None, None
        self.scale = 1

    def _get_area_contribution(self, migration_matrix, effective_area):
        """
        Get area contribution to each bin in migration matrix

        :param migration_matrix: ndarray
            Migration matrix (true energy, reconstructed energy)
        :param effective_area: ndarray
            Effective area (true energy)
        :return: ndarray
            Area contribution to migration matrix bins
        """

        contributions = migration_matrix * effective_area[..., np.newaxis]

        return contributions

    def _get_expected_counts(self, fit_function, migration_matrix, effective_area,
                             **kwargs):
        """
        Get expected number of counts in reconstructed energy for a given spectrum IRF set

        :param fit_function: function pointer
            Function defining spectrum
        :param function_values: dict
            Spectrum function parameters
        :param migration_matrix: ndarray
            Migration matrix (true energy, reconstructed energy)
        :param effective_area: ndarray
            Effective area (true energy)
        :return: ndarray
            Expected counts (reconstructed energy)
        """

        # Get bin width (not in log space) for performing a simple integral later
        el = self.true_energy_bins[0] - np.diff(self.true_energy_bins)[0]
        eu = self.true_energy_bins[-1] + np.diff(self.true_energy_bins)[-1]
        integral_bins = np.logspace(el, eu, (self.true_energy_bins.shape[0] * 100) + 1)
        el = integral_bins[:-1]
        eu = integral_bins[1:]

        energy_bin_width = eu - el

        # Get area contribution matrix
        area_matrix = self._get_area_contribution(migration_matrix, effective_area)

        flux = fit_function(integral_bins,**kwargs)
        # Multiply by source flux
        flux = (eu - el) * (flux[:-1] + flux[1:])/2
        flux = np.sum(flux.reshape(self.true_energy_bins.shape[0], 100), axis=1)
        #print(flux.shape)

        area_matrix *= flux[..., np.newaxis]
        #print fit_function(np.power(10, self.true_energy_bins),
        #                            ** kwargs)
        # Multiply by energy bin width
        #area_matrix *= energy_bin_width[..., np.newaxis]
        # Multiply by simulated area in CORSIKA
        area_matrix *= self.simulated_angular_area

        # Return the sum along the reconstructed energy axis
        return np.sum(area_matrix, axis=0)

    def get_expected_electron_counts(self, fit_function, time, **kwargs):
        """
        Get expected number of electron counts in reconstructed energy

        :param fit_function: function pointer
            Function defining spectrum
        :param function_values: dict
            Spectrum function parameters
        :param time: float
            Observation time (seconds)
        :return: ndarray
            Expected counts (reconstructed energy)
        """

        return self._get_expected_counts(fit_function,
                                         self.electron_migration_matrix,
                                         self.electron_effective_area, **kwargs) * time

    def get_expected_proton_counts(self, fit_function, time, **kwargs):
        """
         Get expected number of proton counts in reconstructed energy

        :param fit_function: function pointer
            Function defining spectrum
        :param function_values: dict
            Spectrum function parameters
        :param time: float
            Observation time (seconds)
        :return: ndarray
            Expected counts (reconstructed energy)
        """
        if self.proton_migration_matrix is None or self.proton_effective_area is None:
            return 0

        return self._get_expected_counts(fit_function,
                                         self.proton_migration_matrix,
                                         self.proton_effective_area, **kwargs) * time

    def _get_poisson_likelihood(self, **kwargs):
        """
        Get the Poissonian likelihood of the current spectral model, in comparison to
        the simulated classifier distribution

        :return: float
            Likelihood of the current spectral model
        """

        # Generate the expected number of electron counts given the current spectral
        # models
        expected_electron_counts = self.get_expected_electron_counts(
            self.electron_fit_function, self.time, **kwargs) * self.scale

        # And do the same for protone if we are fitting a model
        if self.proton_fit_function is not None:
            expected_proton_counts = self.get_expected_proton_counts(
                self.proton_fit_function, self.time, **kwargs)
        else:
            # If not we assume all the non-electron counts in the bin are protons and
            # assume this normalisation
            expected_proton_counts = np.sum(self.total_distribution, axis=1) - \
                                     expected_electron_counts

        # Normalise the templates and add them
        estimated_distribution = self.electron_template * \
                                 expected_electron_counts[..., np.newaxis]
        estimated_distribution += self.proton_template * \
                                  expected_proton_counts[..., np.newaxis]
        estimated_distribution[estimated_distribution<1e-10] = 1e-10
        estimated_distribution[np.isnan(estimated_distribution)] = 0

        # Get log likelihood of model (in chi square form)
        likelihood = -2 * poisson.logpmf(self.total_distribution.astype(np.int),
                                         estimated_distribution)

        likelihood_gaus = -2 * norm.logpdf(self.total_distribution,
                                           estimated_distribution,
                                           np.sqrt(self.total_distribution))

        likelihood[self.total_distribution > 100] = \
            likelihood_gaus[self.total_distribution > 100]


        likelihood = np.sum(likelihood, axis=1)
        if self.energy_range:
            energy_range = np.log10(np.array(self.energy_range))
            energy_range = np.logical_and(self.reco_energy_bins > energy_range[0],
                                          self.reco_energy_bins < energy_range[1])

            likelihood = likelihood[energy_range]
        # Return the sum
        return np.sum(likelihood)

    def _get_poisson_likelihood_minimise(self, x):
        """
        Wrapper around _get_poisson_likelihood, to work with scipy style minimisation.
        Get the Poissonian likelihood of the current spectral model, in comparison to
        the simulated classifier distribution

        :param x: ndarray
            Model parameters
        :return: float
            Likelihood of the current spectral model

        """

        fit_parameters = dict(list(zip(self.variable_names, x)))

        return self._get_poisson_likelihood(**fit_parameters)

    def get_likelihood(self, parameter_distribution, time,
                      fit_function_electrons, parameters_electron,
                      fit_function_proton=None, parameters_proton=None,
                      energy_range=None):
        """
        Get the likelihood of a single spectral model for a give classifier distribution

        :param parameter_distribution: ndarray
            Simuated classifier distribution (classifier, reconstructed energy)
        :param time: float
            Observtaion time (seconds)
        :param fit_function_electrons: function pointer
            Electron spectrum fit function
        :param parameters_electron:dict
            Electron spectral parameters
        :param fit_function_proton: function pointer
            Proton spectrum fit function
        :param parameters_proton: dict
            Proton spectral parameters
        :return: float
            Likelihood of the spectral model
        """

        # Load everything into the class
        self.electron_fit_function = fit_function_electrons
        self.proton_fit_function = fit_function_proton
        self.total_distribution = parameter_distribution
        self.time = time
        self.energy_range = energy_range

        # Concatenate the two dictionaries into one model
        fit_parameters = parameters_electron
        if parameters_proton is not None:
            fit_parameters.update(parameters_proton)

        self.variable_names = list(fit_parameters.keys())
        fit_parameters = list(fit_parameters.values())
        x = []
        for p in fit_parameters:
            try:
                x.append(p[0])
            except:
                x.append(p)

        like = self._get_poisson_likelihood_minimise(x)

        return like

    def get_spectral_points(self, parameter_distribution, time,
                            fit_function_electrons, parameters_electron,
                            fit_function_proton=None, parameters_proton=None):
        """

        :param parameter_distribution: ndarray
            Simuated classifier distribution (classifier, reconstructed energy)
        :param time: float
            Observation time (seconds)
        """


        # Load everything into the class
        self.electron_fit_function = fit_function_electrons
        self.proton_fit_function = fit_function_proton
        self.total_distribution = parameter_distribution
        self.time = time

        # Concatenate the two dictionaries into one model
        fit_parameters = parameters_electron
        if parameters_proton is not None:
            fit_parameters.update(parameters_proton)

        # Copy staring values and limits into arrays
        self.variable_names = list(fit_parameters.keys())
        fit_parameters = list(fit_parameters.values())

        x, limits = [], []
        for p in fit_parameters:
            try:
                x.append(p[0])
                limits.append(p[1])
            except:
                x.append(p)

        value_points, error_points = [], []
        for energy_bin in self.reco_energy_bins:
            range = (np.power(10, energy_bin-0.001),
                     np.power(10, energy_bin+0.001))
            self.energy_range = range

            def scaled_spec(scale):
                self.scale = scale
                return self._get_poisson_likelihood_minimise(x)

            try:
                energy_bin = np.power(10, energy_bin)

                # Perform migrad minimisation
                minimised = minimize(scaled_spec, np.array([np.random.normal(1.0, 0.1)]),
                                     bounds=np.array([[0, 3]]))
                #print(minimised)
                fit_value = minimised["x"][0]
                likelihood = minimised["fun"]
                minuit = minimised["minuit"]

                # Get the errors from MINOS
                try:
                    minuit.minos()
                    errors = np.abs(list(minuit.get_merrors().values())[0]["lower"]), \
                             np.abs(list(minuit.get_merrors().values())[0]["upper"])

                except RuntimeError:
                    errors = (np.nan, np.nan)

                flux = fit_value * fit_function_electrons(energy_bin,
                                                         **parameters_electron)
                val_low = errors[0] * flux
                val_high = errors[1] * flux

                value_points.append(flux)
                error_points.append((val_low, val_high))

            except RuntimeError:
                value_points.append(np.nan)
                error_points.append((np.nan, np.nan))
        self.scale = 1

        return np.power(10,self.reco_energy_bins), np.array(value_points), \
               np.array(error_points).T

    def fit_model(self, parameter_distribution, time,
                  fit_function_electrons, parameters_electron,
                  fit_function_proton=None, parameters_proton=None,
                  energy_range=None):
        """
        Perform a maximum likelihood fit of a given spectral model and return the best
        fit parameters, errors and likelihood

        :param parameter_distribution: ndarray
            Simuated classifier distribution (classifier, reconstructed energy)
        :param time: float
            Observtaion time (seconds)
        :param fit_function_electrons: function pointer
            Electron spectrum fit function
        :param parameters_electron:dict
            Electron spectral parameters and limits
        :param fit_function_proton: function pointer
            Proton spectrum fit function
        :param parameters_proton: dict
            Proton spectral parameters and limits
        :return: dict, dict, float
            Best fit values, fist errors, likelihood at fitted position
        """

        # Load everything into the class
        self.electron_fit_function = fit_function_electrons
        self.proton_fit_function = fit_function_proton
        self.total_distribution = parameter_distribution
        self.time = time
        self.energy_range = energy_range
        self.scale = 1

        # Concatenate the two dictionaries into one model
        fit_parameters = parameters_electron
        if parameters_proton is not None:
            fit_parameters.update(parameters_proton)

        # Copy staring values and limits into arrays
        self.variable_names = list(fit_parameters.keys())
        fit_parameters = list(fit_parameters.values())
        x, limits = [], []
        for p in fit_parameters:
            try:
                x.append(p[0])
                limits.append(p[1])
            except:
                x.append(p)

        try:
            # Perform migrad minimisation
            minimised = minimize(self._get_poisson_likelihood_minimise, np.array(x),
                                 bounds=np.array(limits))
        except RuntimeError:
            return np.nan, np.nan, np.nan

        # Get the fit values
        fit_values = dict(list(zip(self.variable_names, list(minimised["x"]))))
        likelihood = minimised["fun"]
        minuit = minimised["minuit"]

        # Get the errors from MINOS
        fit_errors = dict()
        for key in self.variable_names:
            fit_errors[key] = (np.nan, np.nan)

        if False:
            try:
                minuit.minos()
                minos_errors = minuit.get_merrors()

                for x, key in zip(minos_errors, self.variable_names):
                    x = minos_errors[x]
                    fit_errors[key] = (np.abs(x["lower"]), np.abs(x["upper"]))

            except RuntimeError:
                # If we fail fill with NaN
                for key in self.variable_names:
                    fit_errors[key] = (np.nan, np.nan)

        return fit_values, fit_errors, likelihood
