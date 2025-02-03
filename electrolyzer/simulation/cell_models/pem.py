"""
This module defines a Hydrogen Electrolyzer Cell
"""

# TODOs
# * refine calcCellVoltage(); compare with alkaline models
# * refine convertACtoDC(); compare with empirical ESIF model
# * refine calcFaradaicEfficiency(); compare with other model
# * add a separate script to show results

import numpy as np
from attrs import define
from scipy.constants import R, physical_constants, convert_temperature

from electrolyzer.tools.type_dec import FromDictMixin


def PEM_electrolyzer_model(X, a, b, c, d, e, f):
    """
    Given a power input (kW), temperature (C), and set of coefficients, returns
    current (A).  Coefficients can be determined using non-linear least squares
    fit (see `Stack.create_polarization`).
    """
    P, T = X
    I = a * (P**2) + b * T**2 + c * P * T + d * P + e * T + f

    return I


# Constants #
#############
F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]
P_ATMO, _, _ = physical_constants["standard atmosphere"]  # Pa
P_STD, _, _ = physical_constants["standard-state pressure"]  # Pa (1bar)


@define
class PEMCell(FromDictMixin):
    # Chemical Params #
    ###################

    cell_area: float
    turndown_ratio: float
    max_current_density: float

    # If we rework this class to be even more generic, we can have these be specified
    # as configuration params

    # number of electrons transferred in reaction
    n: int = 2

    gibbs: float = 237.24e3  # Gibbs Energy of global reaction (J/mol)
    M: float = 2.016  # molecular weight [g/mol]
    lhv: float = 33.33  # lower heating value of H2 [kWh/kg]
    hhv: float = 39.41  # higher heating value of H2 [kWh/kg]

    def calc_reversible_voltage(self):
        """
        Calculates reversible cell potential at standard state.
        """
        return self.gibbs / (self.n * F)

    def calc_open_circuit_voltage(self, temperature):
        """Calculates open circuit voltage using the Nernst equation."""
        T_K = convert_temperature([temperature], "C", "K")[0]
        E_rev_0 = self.calc_reversible_voltage()
        p_anode = P_ATMO  # (Pa) assumed atmo
        p_cathode = 30 * P_STD  # (Pa) 30 bars

        # noqa: E501
        # Arden Buck equation T=C, https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#vapor-pressure-formulas # noqa
        # Reasonable at temperatures between 0-100C
        p_h2O_sat = (
            0.61121
            * np.exp(
                (18.678 - (temperature / 234.5))
                * (temperature / (257.14 + temperature))
            )
        ) * 1e3  # (Pa)

        # Dalton's Law to find partial pressure of reactants at each electrode.
        p_h2 = p_cathode - p_h2O_sat
        p_o2 = p_anode - p_h2O_sat

        # General Nernst equation, 10.1016/j.ijhydene.2017.03.046
        E_cell = E_rev_0 + ((R * T_K) / (self.n * F)) * (
            np.log((p_h2 / P_ATMO) * np.sqrt(p_o2 / P_ATMO))
        )

        return E_cell

    def calc_activation_overpotential(self, i, temperature):
        """
        Calculates activation overpotential for a given current density and temperature.
        """
        T_K = convert_temperature([temperature], "C", "K")[0]
        # Option 1:

        # constants below assumed from https://www.sciencedirect.com/science/article/pii/S0360319916318341?via%3Dihub # noqa

        # TODO: updated with realistic anode temperature? 70-80 C nominal operating
        # temperature 58C
        T_anode = T_K

        T_cathode = T_K  # TODO: updated with realistic anode temperature?

        # anode charge transfer coefficient TODO: is this a realistic value?
        alpha_a = 2

        # cathode charge transfer coefficient TODO: is this a realistic value?
        alpha_c = 0.5

        # anode exchange current density TODO: update to be f(T)?
        i_0_a = 2e-7

        # cathode exchange current density TODO: update to be f(T)?
        i_0_c = 2e-3

        # derived from Butler-Volmer eqs
        V_act_a = ((R * T_anode) / (alpha_a * F)) * np.arcsinh(i / (2 * i_0_a))
        V_act_c = ((R * T_cathode) / (alpha_c * F)) * np.arcsinh(i / (2 * i_0_c))

        # alternate equations for Activation overpotential
        # Option 2: Dakota: I believe this may be more accurate, found more
        # frequently in lit review
        # https://www.sciencedirect.com/science/article/pii/S0360319918309017

        # z_a = 4 # stoichiometric coefficient of electrons transferred at anode
        # z_c = 2 # stoichometric coefficient of electrons transferred at cathode
        # i_0_a = 10**(-9) # anode exchange current density TODO: update to be f(T)?
        # i_0_c = 10**(-3) # cathode exchange current density TODO: update to be f(T)?

        # V_act_a = ((R*T_anode)/(alpha_a*z_a*F)) * np.log(i/i_0_a)
        # V_act_c = ((R*T_cathode)/(alpha_c*z_c*F)) * np.log(i/i_0_c)

        return V_act_a, V_act_c

    def calc_ohmic_overpotential(self, i, temperature):
        """
        Calculates Ohmic overpotential for a given current density and temperature.
        """
        T_K = convert_temperature([temperature], "C", "K")[0]

        # pulled from https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub # noqa
        # TODO: pulled from empirical data, is there a better eq?
        lambda_nafion = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        t_nafion = 0.02  # (cm) confirmed that membrane thickness is <0.02.

        # TODO: confirm with Nel, is there a better eq?
        sigma_nafion = ((0.005139 * lambda_nafion) - 0.00326) * np.exp(
            1268 * ((1 / 303) - (1 / T_K))
        )
        R_ohmic_ionic = t_nafion / sigma_nafion

        # TODO: confirm realistic value with Nel https://www.sciencedirect.com/science/article/pii/S0378775315001901 # noqa
        R_ohmic_elec = 50e-3

        # Alternate R_ohmic_elec from https://www.sciencedirect.com/science/article/pii/S0360319918309017 # noqa
        # rho =  (ohm*m) material resistivity
        # l_path = (m) length of electron path
        # A_path = (m2) cross-sectional area of conductor path
        # R_ohmic_elec = ((rho*l_path)/A_path)

        V_ohmic = i * (R_ohmic_elec + R_ohmic_ionic)

        return V_ohmic

    def calc_concentration_overpotential(self):
        # TODO: complete this section
        # Option 1:
        # https://www.sciencedirect.com/science/article/pii/S0360319918309017
        # equations governing concentration losses / diffusion are pretty complex;
        # hoping to get an approx value or eqs from Kaz / Nel for concentration of O2
        # and H2 at electrodes else can add in equations from this paper or others to
        # get into diffusion.
        # C_an_mem_o2 = TODO: complete with equations or can we use approx values?
        # C_an_mem_o2_0 = TODO: complete with equations or can we use approx values?
        # C_cat_mem_h2 = TODO: complete with equations or can we use approx values?
        # C_cat_mem_h2_0 = TODO: complete with equations or can we use approx values?

        # V_con = ((((R*T_K)/(4*F))*np.log(C_an_mem_o2/C_an_mem_o2_0)) + (((R*T_K)/(4*F))*np.log(C_cat_mem_h2/C_cat_mem_h2_0))) # noqa

        # Option 2:
        # PEM Fuel Cell Modeling and simulation using MATLAB ISBN 978-0-12-374259-9
        # (saved in H2@scale teams>Lit Review>Fuel Cells folder)

        # Similar relationship with -log(1-i/i_L) found here
        # https://doi.org/10.1016/j.jclepro.2020.121184

        # i_L = #limiting current density TODO: get value or eq from Nel / Kaz?
        # V_con = ((R*T_K)/(self.n*F))*np.log((i_L/(i_L-i)))
        return 0

    def calc_overpotentials(self, i, temperature):
        """
        Calculates overpotentials for a given current density and temperature.
        """
        V_act_a, V_act_c = self.calc_activation_overpotential(i, temperature)
        V_ohm = self.calc_ohmic_overpotential(i, temperature)
        V_conc = self.calc_concentration_overpotential()

        return (V_act_a, V_act_c, V_ohm, V_conc)

    def calc_cell_voltage(self, I, temperature):
        """
        I [Adc]: current
        return :: V_cell [Vdc/cell]: cell voltage
        """
        i = I / self.cell_area  # current density, A/cm^2

        E_cell = self.calc_open_circuit_voltage(temperature)
        overpotentials = self.calc_overpotentials(i, temperature)

        V_cell = E_cell + sum(overpotentials)

        return V_cell

    # ------------------------------------------------------------
    # Post H2 production
    # ------------------------------------------------------------
    def calc_faradaic_efficiency(self, T_C, I):
        """
        I [A]: current
        T_C [C]: cell temperature (currently unused)
        return :: eta_F [-]: Faraday's efficiency
        Reference: https://res.mdpi.com/d_attachment/energies/energies-13-04792/article_deploy/energies-13-04792-v2.pdf
        """  # noqa
        f_1 = 250  # (mA2/cm4)
        f_2 = 0.996
        I *= 1000

        eta_F = (
            ((I / self.cell_area) ** 2) / (f_1 + ((I / self.cell_area) ** 2))
        ) * f_2

        return eta_F

    def calc_mass_flow_rate(self, T_C, Idc, dryer_loss=6.5):
        """
        Idc [A]: stack current
        dryer_loss [%]: loss of drying H2
        T_C [C]: cell temperature (currently unused)
        return :: mfr [kg/s]: mass flow rate
        """
        eta_F = self.calc_faradaic_efficiency(T_C, Idc)
        mfr = eta_F * Idc * self.M / (self.n * F) * (1 - dryer_loss / 100.0)  # [g/s]
        # mfr = mfr / 1000. * 3600. # [kg/h]
        mfr = mfr / 1e3  # [kg/s]
        return mfr
