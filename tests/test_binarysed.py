import pytest
import numpy as np
import pandas as pd
from binarysed import SED, stefan_boltzmann_luminosity, calculate_z, calculate_effective_extinction

@pytest.fixture
def sed_data():
    return {
        "fluxes": np.array([1.0e-10, 2.0e-10, 3.0e-10]),
        "flux_errs": np.array([1.0e-11, 2.0e-11, 3.0e-11]),
        "wavelengths": np.array([1000, 2000, 3000]),
        "filters": ["filter1", "filter2", "filter3"],
        "RA": 100.0,
        "DEC": -30.0,
        "dist": 100.0  # in pc
    }

@pytest.fixture
def sed_instance(sed_data):
    return SED(sed_data)

def test_get_obs_fluxes(sed_instance):
    assert np.allclose(sed_instance.get_obs_fluxes(), [1.0e-10, 2.0e-10, 3.0e-10])

def test_get_obs_flux_errs(sed_instance):
    assert np.allclose(sed_instance.get_obs_flux_errs(), [1.0e-11, 2.0e-11, 3.0e-11])

def test_get_obs_wavelengths(sed_instance):
    assert np.allclose(sed_instance.get_obs_wavelengths(), [1000, 2000, 3000])

def test_get_obs_filters(sed_instance):
    assert sed_instance.get_obs_filters() == ["filter1", "filter2", "filter3"]

def test_get_ra(sed_instance):
    assert sed_instance.get_ra() == 100.0

def test_get_dec(sed_instance):
    assert sed_instance.get_dec() == -30.0

def test_get_dist(sed_instance):
    assert sed_instance.get_dist() == 100.0

def test_create_intrinsic_sed(sed_instance):
    # Define parameters for the test
    teff = 6000  # K
    logg = 4.0  # cm/s^2
    lum = 1.0  # Lsun
    Z = 0.02  # solar metallicity
    dist = 100  # pc
    wavelengths = np.array([1000, 2000, 3000])  # Angstroms

    # Create intrinsic SED
    sed_df = sed_instance.create_intrinsic_sed(teff, logg, lum, Z, dist, wavelengths)

    # Assert that the dataframe has the correct structure and values
    assert "wavelength" in sed_df.columns
    assert "flux" in sed_df.columns

def test_create_apparent_sed(sed_instance):
    # Define parameters for the test
    wavelengths = np.array([1000, 2000, 3000])  # Angstroms
    teff1 = 6000  # K
    teff2 = 5500  # K
    requiv1 = 1.0  # Rsun
    requiv2 = 0.9  # Rsun
    logg1 = 4.0  # cm/s^2
    logg2 = 4.2  # cm/s^2

    # Create apparent SED
    apparent_flux, _ = sed_instance.create_apparent_sed(wavelengths, teff1, teff2, requiv1, requiv2, logg1, logg2)

    # Assert that the apparent flux is correctly calculated (basic check to see if values are less than original flux)
    assert np.all(apparent_flux <= 1.0e-9)  # Example threshold, adjust as needed

def test_calculate_z():
    dist = 100  # pc
    galactic_latitude = 30  # degrees
    z = calculate_z(dist, galactic_latitude)

    assert np.isclose(z, 50.0, atol=0.1)  # Expected height above the plane

def test_calculate_effective_extinction():
    ebv_sfd = 0.1
    z = 100  # pc
    h = 130  # pc, default scale height

    ebv_eff = calculate_effective_extinction(ebv_sfd, z, h)
    assert ebv_eff < ebv_sfd

def test_stefan_boltzmann_luminosity():
    radius = 1.0  # Rsun
    teff = 5800  # K

    luminosity = stefan_boltzmann_luminosity(radius, teff)

    # The expected luminosity of the Sun
    assert np.isclose(luminosity, 1.0, atol=0.1)  # Lsun

def test_plot_sed_data(sed_instance):
    # Test if the plot function runs without errors
    try:
        sed_instance.plot_sed_data()
    except Exception:
        pytest.fail("plot_sed_data() raised Exception unexpectedly!")
