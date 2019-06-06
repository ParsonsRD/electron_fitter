from root_numpy import hist2array, tree2array
from rootpy.io import root_open
from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import quad
from numpy.random import normal as random_norm

__all__ = ['make_irfs_from_root']


def array2d_from_histogram(hist):
    """
    Get numpy array from 2D ROOT histograms

    :param hist: hist2d
        2D ROOT histogram
    :return: (bin centres, bin content)
    """
    hist, (x, y) = hist2array(hist, copy=True, return_edges=True)

    # Convert bin edges to bin centres
    x = get_bin_centre(x)
    y = get_bin_centre(y)

    return (x, y), hist


def array_from_histogram(hist):
    """
    Get numpy array from 1D ROOT histograms

    :param hist: hist1d
        1D ROOT histogram
    :return: (bin centres, bin content)
    """
    hist, x = hist2array(hist, copy=True, return_edges=True)
    x = get_bin_centre(x[0])

    return x, hist


def get_bin_centre(bin_edges):
    """
    Get bins centre from bin edges

    :param bin_edges: ndarray
        Bin edges
    :return: ndarray
        Bin centres
    """
    return bin_edges[:-1] + (np.diff(bin_edges) / 2)


def make_irfs_from_root(filename, spectral_function, spectral_parameters=None,
                        telescope_multiplicity=2, angular_cut=4,
                        energy_range=(-3, 3), energy_bins=(1200, 60),
                        zeta_range=(0.2, 1), zeta_bins=20,
                        simulated_radius=3000, simulated_angular_cone=10,
                        energy_resolution=None):
    """
    Creates the required instrument response functions from the input ROOT tree

    :param filename: str
        Name of input ROOT file
    :param spectral_function: function_ptr
        Pointer to spectrum function
    :param spectral_parameters: dict
        Dictionary of spectrum function inputs
    :param telescope_multiplicity: int
        Minimum telescope multiplicity
    :param angular_cut: float
        Maximum reconstructed event offset used
    :param energy_range: (float, float)
        Maximum and minimum energies to use log10(TeV)
    :param energy_bins: (int, int)
        Number of energy bins (True energy, Reconstructed energy)
    :param zeta_range: (float, float)
        Range of zeta parameter used in templates
    :param zeta_bins: int
        Number of Zeta bins
    :param simulated_radius: float
        Maximum event radius simulated
    :param simulated_angular_cone: float
        Maximum angular deviation simulated
    :return: (effective area, migration matrix, classifier distribution)

    """
    # Open up our root file and get the simulated event histogram
    input_file = root_open(filename)
    simulated_energy, event_number = array2d_from_histogram(input_file.Get("1"))
    event_number = np.sum(event_number, axis=0)
    simulated_energy = simulated_energy[1]

    simulated_events = interp1d(simulated_energy, event_number, kind="nearest")

    # Generate the angular spread and core spread areas used in the simulations
    simulated_angular_area = 2 * np.pi * (1 - np.cos(np.deg2rad(simulated_angular_cone)))
    simulated_area = np.pi * simulated_radius * simulated_radius

    # Dump the TTree into a numpy array
    event_array = tree2array(input_file.out, branches=["EnergyTrue", "EnergyRec",
                                                       "AngOffsetTrue", "AngOffsetRec",
                                                       "NImg", "eta"])

    # Make selection cuts on telescope multiplicity and reconstructed angular offset
    selection_cuts = np.logical_and(event_array["AngOffsetRec"] < angular_cut,
                                    event_array["NImg"] > telescope_multiplicity - 1)
    selection_cuts = np.logical_and(selection_cuts, event_array["eta"] > zeta_range[0])


    # Put these in log units as we'll be using them like that later
    mc_energy = (event_array["EnergyTrue"][selection_cuts])
    if energy_resolution:
        scatter = random_norm(loc=1, scale=energy_resolution, size=mc_energy.shape)
        reconstructed_energy = np.log10(event_array["EnergyTrue"][selection_cuts] *
                                        scatter)
    else:
        reconstructed_energy = (event_array["EnergyRec"][selection_cuts])

    eta = event_array["eta"][selection_cuts]


    # Effective area start my maing an event wise weight for each event, based on the
    # number of events of that energy simulated and the area spread
    event_area = simulated_area / simulated_events(mc_energy)
    # Fill up our histogram with MC energy and weight
    effective_area = np.histogram(mc_energy, weights=event_area, range=energy_range,
                                  bins=energy_bins[0])
    effective_area_bins = get_bin_centre(effective_area[1])
    # We have to add a correction factor based in the differences in widths between the
    # simulated event histogram and out effective area histogram (assume fixed bin size)
    area_binning_factor = np.diff(simulated_energy)[0]/np.diff(effective_area_bins)[0]
    effective_area = effective_area_bins, effective_area[0] * area_binning_factor

    def spectral_integration_function(energy):
        return spectral_function(energy, **spectral_parameters)

    integrated_spectral_bins = np.zeros_like(event_number)
    bin_width = np.diff(simulated_energy)[0]/2
    if not spectral_parameters:
        spectral_parameters = {}

    for energy_bin in range(simulated_energy.shape[0]):
        lower_energy = np.power(10, simulated_energy[energy_bin] - bin_width)
        upper_energy = np.power(10, simulated_energy[energy_bin] + bin_width)

        integrated_spectral_bins[energy_bin] = \
            quad(spectral_integration_function, lower_energy, upper_energy)[0]

    integrated_spectral_bins = interp1d(simulated_energy, integrated_spectral_bins,
                                        kind="nearest")

    event_weight = (simulated_area * integrated_spectral_bins(mc_energy) *
                    simulated_angular_area) / simulated_events(mc_energy)

    # Next up make the migration matrix, star by filling a histogram with MC energy vs
    # reconstructed energy
    migration_matrix = np.histogram2d(mc_energy, reconstructed_energy,
                                      range=(energy_range, energy_range),
                                      bins=energy_bins, weights=event_weight)

    # Normalise the sum of MC energy columns to equal 1
    normalised_migration_matrix = migration_matrix[0] / \
                                  np.sum(migration_matrix[0], axis=1)[..., np.newaxis]
    normalised_migration_matrix[np.isnan(normalised_migration_matrix)] = 0
    # Make out output object of bin centres and the matrix
    migration_matrix = (get_bin_centre(migration_matrix[1]),
                        get_bin_centre(migration_matrix[2])), normalised_migration_matrix


    classifier_distribution = np.histogram2d(reconstructed_energy, eta,
                                             weights=event_weight,
                                             bins=(energy_bins[1], zeta_bins),
                                             range=(energy_range, zeta_range))

    classifier_distribution = ((get_bin_centre(classifier_distribution[1]),
                               get_bin_centre(classifier_distribution[2])),
                               classifier_distribution[0])
    input_file.Close()

    return effective_area, migration_matrix, classifier_distribution

