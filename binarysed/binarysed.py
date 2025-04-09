import numpy as np
import pandas as pd
import math
import os
from pystellibs import BaSeL, Kurucz
from extinction import ccm89, apply, remove
import pyphot
# from dustmaps.sfd import SFDQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib import pyplot as plt

# Load kurucz library for SED; Castelli and Kurucz 2004 or ATLAS9
kurucz = Kurucz()
# Provide relevant filter names from the pyphot library
filters = [
    "GALEX_FUV",
    "GALEX_NUV",
    "PS1_g",
    "PS1_i",
    "PS1_r",
    "PS1_y",
    "PS1_z",
    "TYCHO_B_MvB",
    "TYCHO_V_MvB",
    "SkyMapper_g",
    "SkyMapper_i",
    "SkyMapper_r",
    "SkyMapper_u",
    "SkyMapper_v",
    "SkyMapper_z",
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "WISE_RSR_W1",
    "WISE_RSR_W2",
    "WISE_RSR_W3",
    "WISE_RSR_W4",
]
# Load the pyphot library
lib = pyphot.get_library()


class SED:
    """
    Class to handle the SED of a star.

    Attributes
    ----------
    sed_dict : dict
        Dictionary containing the SED data of the star.

    Methods
    -------
    get_obs_fluxes()
        Return the observed fluxes of the star.
    get_obs_flux_errs()
        Return the observed flux errors of the star.
    get_obs_wavelengths()
        Return the observed wavelengths of the star.
    get_obs_filters()
        Return the observed filters of the star.
    get_ra()
        Return the right ascension of the star.
    get_dec()
        Return the declination of the star.
    get_dist()
        Return the distance to the star.
    create_intrinsic_sed(teff, logg, lum, Z, dist, wavelengths)
        Create the intrinsic SED of a star at a given distance.
    create_apparent_sed(wavelengths, teff1, teff2, requiv1, requiv2, logg1, logg2, select_wavelengths=False)
        Create the apparent SED of a binary star system at a given distance.
    plot_sed_data()
        Plot the observed SED of the star.
    """

    def __init__(self, sed_dict):
        self.sed_dict = sed_dict
        # self.sfd = SFDQuery()
        self.edenhofer = Edenhofer2023Query(integrated=True)
        binarysed_dir = os.path.dirname(os.path.abspath(__file__))
        extinction_curve_path = os.path.join(binarysed_dir, "docs", "extinction_curve.txt")
        self.extinction_curve = pd.read_csv(extinction_curve_path, delim_whitespace=True, 
                            names=["wavelength", "extinction_curve"], skiprows=1)
        print(f"wavelengths: {self.extinction_curve['wavelength']}")


    def get_obs_fluxes(self):
        return self.sed_dict["fluxes"]

    def get_obs_flux_errs(self):
        return self.sed_dict["flux_errs"]

    def get_obs_wavelengths(self):
        return self.sed_dict["wavelengths"]

    def get_obs_filters(self):
        return self.sed_dict["filters"]

    def get_ra(self):
        return self.sed_dict["RA"]

    def get_dec(self):
        return self.sed_dict["DEC"]

    def get_dist(self):
        return self.sed_dict["dist"]

    def create_intrinsic_sed(self, teff, logg, lum, Z, dist, wavelengths):
        """
        Create the intrinsic SED of a star at a given distance

        Parameters
        ----------
        teff : float
            Effective temperature of the star in K.
        logg : float
            Surface gravity of the star in log10(cm/s^2).
        lum : float
            Luminosity of the star in Lsun.
        Z : float
            Metallicity of the star.
        dist : float
            Distance to the star in pc.
        wavelengths : list
            List of wavelengths at which to calculate the SED.

        Returns
        -------
        sed_df : pandas.DataFrame
            DataFrame containing the intrinsic SED of the star at the given distance.

        """

        # Convert distance from pc to cm
        dist = dist * 3.086e18
        # Convert luminosity from Lsun to flux in erg/s
        fluxconv = 4 * np.pi * dist * dist
        # Store inputs for generating the stellar spectrum
        ap = (np.log10(teff), logg, np.log10(lum), Z)
        # Generate stellar spectrum using Kurucz library from pystellibs
        kurucz_sed = kurucz.generate_stellar_spectrum(*ap)

        # Get the valid indices of the Kurucz SED
        nonzero = np.where(kurucz_sed > 0)[0]
        # Create a DataFrame of the Kurucz SED and add the desired model wavelengths
        sed = pd.DataFrame(
            {
                "wavelength": np.append(kurucz._wavelength[nonzero], wavelengths),
                "flux": np.append(
                    kurucz_sed[nonzero] / fluxconv, np.full((len(wavelengths),), np.nan)
                ),
            }
        )
        
        # Sort the SED by wavelength
        sed_interp = sed.sort_values(by="wavelength")

        # Select the flux values at only the specified wavelengths
        for wave in wavelengths:
            # The interpolation is done in log-log space
            # Interpolate over the Kurucz SED to get the flux values at the desired model wavelengths 
            # (rather than just the Kurucz wavelengths)
            flux_interpolated = np.interp(np.log10(wave), np.log10(sed["wavelength"][: -len(wavelengths)]),
                                          np.log10(sed["flux"][: -len(wavelengths)]))
            # Store the interpolated flux values in the DataFrame after converting back to linear space
            sed_interp.loc[sed_interp["wavelength"] == wave, "flux"] = (
                10**flux_interpolated
            )

        wavelength_range = (sed_interp["wavelength"] > 1000) & (sed_interp["wavelength"] < 3e5)
        # Create a DataFrame of the interpolated SED
        sed_df = pd.DataFrame(
            {"wavelength": sed_interp["wavelength"].loc[wavelength_range], "flux": sed_interp["flux"][wavelength_range]}
        )

        # Return the intrinsic SED based on the given distance (wavelength in angstroms, flux in erg/cm2/s/AA)
        return sed_df

    def create_apparent_sed(
        self,
        wavelengths,
        teff1,
        teff2,
        requiv1,
        requiv2,
        logg1,
        logg2,
        select_wavelengths=False,
    ):
        """
        Create the apparent SED of a binary star system at a given distance.

        Parameters
        ----------
        wavelengths : list
            List of wavelengths at which to calculate the SED.
        teff1 : float
            Effective temperature of the primary star in K.
        teff2 : float
            Effective temperature of the secondary star in K.
        requiv1 : float
            Radius of the primary star in Rsun.
        requiv2 : float
            Radius of the secondary star in Rsun.
        logg1 : float
            Surface gravity of the primary star in log10(cm/s^2).
        logg2 : float
            Surface gravity of the secondary star in log10(cm/s^2).
        select_wavelengths : bool, optional
            If True, only return the SED values at the specified wavelengths.
            Default is False, in which the SED values are returned at the Kurucz wavelengths.

        Returns
        -------
        comb_sed : pandas.DataFrame
            DataFrame containing the apparent SED of the binary star system at the given distance.
        """
        R_V = 3.1
        Z = 0.01
        RA = self.sed_dict["RA"]
        DEC = self.sed_dict["DEC"]
        dist = self.sed_dict["dist"]

        # Calculate the luminosities of the stars using the Stefan-Boltzmann law
        lum1 = stefan_boltzmann_luminosity(requiv1, teff1)
        lum2 = stefan_boltzmann_luminosity(requiv2, teff2)

        # Create the intrinsic SEDs of the two stars at the given distance
        sed_1 = self.create_intrinsic_sed(
            teff1, logg1, lum1, Z, dist, wavelengths
        ).drop_duplicates(subset=["wavelength"])
        sed_2 = self.create_intrinsic_sed(
            teff2, logg2, lum2, Z, dist, wavelengths
        ).drop_duplicates(subset=["wavelength"])

        # Optional: Select only the SED values at the specified wavelengths
        if select_wavelengths:
            sed_1 = sed_1[sed_1["wavelength"].isin(wavelengths)]
            sed_2 = sed_2[sed_2["wavelength"].isin(wavelengths)]

        # Merge the SED dataframes on the wavelength
        merged_seds = pd.merge(sed_1, sed_2, on="wavelength", suffixes=("_1", "_2"))

        # Sum the flux values from both dataframes to get the combined SED for the binary system
        merged_seds["flux"] = merged_seds["flux_1"] + merged_seds["flux_2"]

        # Optional: Group by wavelength and sum the total fluxes because some wavelengths may have multiple entries
        comb_sed = merged_seds.groupby("wavelength").agg({"flux": "sum"}).reset_index()

        # Get the galactic latitude from RA/DEC
        coord = SkyCoord(ra=RA * u.degree, dec=DEC * u.degree, distance=dist * u.pc, frame="icrs")
        galactic_coord = coord.galactic
        # galactic_latitude = galactic_coord.b.degree

        # Get the SFD color excess: E(B-V) from the Schlegel, Finkbeiner, & Davis (1998) dust map
        # ebv_sfd = self.sfd(coord)
        A_ZGR23 = self.edenhofer(coord)
        print(f"A (unitless): {A_ZGR23}")
        # Calculate the height above the Galactic plane
        # z = calculate_z(dist, galactic_latitude)
        # # Calculate the effective extinction E(B-V)_eff based on the height above the Galactic plane and the SFD extinction value
        # ebv_eff = calculate_effective_extinction(ebv_sfd, z)

        # # Check that the effective color excess is less than the SFD color excess, otherwise throw an error
        # if ebv_eff > ebv_sfd:
        #     raise ValueError(
        #         "Effective extinction cannot be greater than SFD extinction."
        #     )

        # # Calculate the effective extinction A_V based on the effective color excess and the extinction law
        # A_V = R_V * ebv_eff

        # Apply extinction (Cardelli, Clayton, & Mathis 1989) to the combined SED to get the reddened apparent SED
        print(f"observed wavelengths/10: {comb_sed.wavelength.values/10}")
        extinction_coefficients = np.interp(comb_sed.wavelength.values/10,
                                            self.extinction_curve["wavelength"],
                                            self.extinction_curve["extinction_curve"])
        print(f"extinction coefficients: {extinction_coefficients}")
        A_lambdas = A_ZGR23 * extinction_coefficients
        print(f"A_lambdas: {A_lambdas}")

        comb_sed["apparent_flux"] = apply(
            A_lambdas, comb_sed["flux"].values
        )

        # Check that all apparent fluxes with reddening are <= the original fluxes, otherwise throw an error
        if not all(comb_sed["apparent_flux"] <= comb_sed["flux"]):
            raise ValueError(
                "Apparent fluxes with reddening cannot be greater than original fluxes."
            )

        # Returns the combined apparent binary SED with reddening
        # Wavelength in angstroms, flux in erg/cm2/s/AA
        if select_wavelengths:
            return comb_sed["apparent_flux"]  # Return model only at desired wavelengths
        else:
            print('Returning apparent SED')
            return (
                comb_sed["apparent_flux"],
                comb_sed["wavelength"],
            )  # Return model at Kurucz wavelengths

    def plot_sed_data(self):
        """
        Plot the observed SED of the star.

        Returns
        -------
        None
        """
        wavelengths = self.sed_dict["wavelengths"]
        fluxes = self.sed_dict["fluxes"]
        flux_errors = self.sed_dict["flux_errs"]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")
        ax.tick_params(
            direction="in", axis="both", which="minor", length=3, width=2, labelsize=18
        )
        ax.tick_params(
            direction="in", axis="both", which="major", length=6, width=2, labelsize=18
        )
        ax.minorticks_on()
        ax.set_xlabel(r"$ \rm Wavelength [nm]$", fontsize=24)
        ax.set_ylabel(r"$\lambda F_\lambda~[\rm erg/cm^2/s]$", fontsize=24)
        ax.errorbar(
            np.array(wavelengths) / 10,
            fluxes * np.array(wavelengths),
            yerr=flux_errors * np.array(wavelengths),
            label="Data",
            marker=".",
            linestyle="None",
            markersize=10,
            c="k",
            zorder=1000,
            capsize=5,
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.vlines(
            wavelengths / 10,
            ymin=10**-14,
            ymax=10**-9,
            linestyle="--",
            color="green",
            alpha=0.5,
            linewidth=0.5,
        )
        plt.ylim(10**-15, 10**-9)
        plt.xlim(10**2, 10**5)
        plt.show()

    def plot_sed_and_model(
        self,
        teff1,
        teff2,
        requiv1,
        requiv2,
        logg1,
        logg2,
        Z=0.01,
        fig=None,
        ax=None,
        savefile=None,
    ):

        wavelengths = np.array(self.sed_dict["wavelengths"])
        fluxes = np.array(self.sed_dict["fluxes"])
        flux_errors = np.array(self.sed_dict["flux_errs"])
        model_fluxes, model_wavelengths = self.create_apparent_sed(
            wavelengths, teff1, teff2, requiv1, requiv2, logg1, logg2
        )

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_ticks_position("both")
            ax.tick_params(
                direction="in",
                axis="both",
                which="minor",
                length=3,
                width=2,
                labelsize=18,
            )
            ax.tick_params(
                direction="in",
                axis="both",
                which="major",
                length=6,
                width=2,
                labelsize=18,
            )
            ax.minorticks_on()
            ax.set_xlabel(r"$ \rm Wavelength [nm]$", fontsize=24)
            ax.set_ylabel(r"$\lambda F_\lambda~[\rm erg/cm^2/s]$", fontsize=24)
            ax.set_yscale("log")
            ax.set_xscale("log")
            for filt in self.sed_dict["filters"]:
                ax.vlines(
                    wavelengths / 10,
                    ymin=10**-14,
                    ymax=10**-9,
                    linestyle="--",
                    color="green",
                    alpha=0.5,
                    linewidth=0.5,
                )
            plt.ylim(10**-15, 10**-9)
            plt.xlim(10**2, 10**5)
            ax.errorbar(
                wavelengths / 10,
                fluxes * np.array(wavelengths),
                yerr=flux_errors * np.array(wavelengths),
                label="Data",
                marker=".",
                linestyle="None",
                markersize=10,
                c="k",
                zorder=1000,
                capsize=5,
            )

        ax.plot(
            np.array(model_wavelengths) / 10,
            model_fluxes * np.array(model_wavelengths),
        )

        return fig, ax


def calculate_z(distance, galactic_latitude):
    """
    Calculate the height above the Galactic plane for a star.

    Parameters:
        distance (float): Distance to the star in parsecs.
        galactic_latitude (float): Galactic latitude of the star in degrees.

    Returns:
        float: Height above the Galactic plane in parsecs.
    """
    # Convert galactic latitude from degrees to radians
    b_rad = np.radians(galactic_latitude)

    # Calculate z
    z = distance * np.sin(b_rad)
    return z


def calculate_effective_extinction(ebv_sfd, z, h=130):
    """
    Calculate the effective extinction E(B-V)_eff for a system based on its height 
    above the Galactic plane and the SFD extinction value.
    (see Equation 1 of https://arxiv.org/pdf/1105.0055)

    Parameters:
        ebv_sfd (float): The SFD extinction towards the system.
        z_i (float): The height of the system above the Galactic plane in parsecs.
        h (float, optional): The exponential scale height of dust in the disk. Defaults to 130 pc.

    Returns:
        float: The effective extinction E(B-V)_eff.
    """
    ebv_eff = ebv_sfd * (1 - np.exp(-np.abs(z) / h))
    return ebv_eff


def stefan_boltzmann_luminosity(radius_in_solar_radii, teff_in_kelvin):
    """
    Calculate the luminosity of a star given its radius and effective temperature.
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
    radius_in_meters = radius_in_solar_radii * 6.96e8  # Convert solar radii to meters
    L_sun = 3.828e26  # Solar luminosity in Watts

    # Calculate the luminosity
    luminosity = 4 * math.pi * radius_in_meters**2 * sigma * teff_in_kelvin**4

    # Convert luminosity to solar luminosities
    luminosity_solar = luminosity / L_sun

    return luminosity_solar
