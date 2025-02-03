"""
This module defines an Alkaline Hydrogen Electrolyzer Cell
"""

import warnings

import numpy as np

# TODO
# * refine calcCellVoltage(); compare with alkaline models
# * refine convertACtoDC(); compare with empirical ESIF model
# * refine calcFaradaicEfficiency(); compare with other model
# * add a separate script to show results
from attrs import field, define
from scipy.constants import R, physical_constants, convert_temperature

from electrolyzer.tools.type_dec import FromDictMixin


warnings.filterwarnings("ignore")
"""
[Oystein Ulleberg, 2003]
    "Modeling of advanced alkaline electrolyzers: a system simulation approach"
    https://www.sciencedirect.com/science/article/pii/S0360319902000332?via%3Dihub

[Gambou, Guilbert,et al 2022]
    "A Comprehensive Survey of Alkaline Eelctrolyzer Modeling: Electrical
    Domain and Specific Electrolyte Conductivity"
    https://www.mdpi.com/1996-1073/15/9/3452

[Haug,Kreitz, 2017]
    "Process modelling of an alkaline water electrolyzer"
    https://www.sciencedirect.com/science/article/pii/S0360319917318633

[Henou, Agbossou, 2014]
    "Simulation tool based on a physics model and an electrical
    analogy for an alkaline electrolyser"
    https://www.sciencedirect.com/science/article/pii/S0378775313017527
    ->cited by [Gambou, Guilbert,et al 2022]
    -> HAS ALL THE VALUES FOR VARIABLES USED IN [Gambou, Guilbert,et al 2022]

[Hammoudi,Henao, 2012]
    "New multi-physics approach for modelling andn design of
    alkaline electrolyzers"
    https://www.sciencedirect.com/science/article/pii/S036031991201590X
    -> Referenced by [Henou, Agbossou, 2014] for theta calculation
        (electrode coverage)
    ->Eqn 44 for bubble stuff
    ->j_lim=300 kA/m^2
    ->includes other efficiency losses
    cites:
    https://www.sciencedirect.com/science/article/pii/S0360128509000598

[Brauns,2021]
"Evaluation of Diaphragms and Membranes as Separators for Alkaline
Water Electrolysis"
    by Jorn Brauns et all 2021. J. Electrochem Soc 168 014510
    https://iopscience.iop.org/article/10.1149/1945-7111/abda57/pdf
    ->good numbers
    ->electrolyte flow rate of 350 mL/min
    ->total electrolyte volume of 10L
    -> has supplementary material (need to checkout)
    ->in "material stability" it mentions stuff about DEGRADATION

NEL Report:
https://www.energy.gov/sites/default/files/2022-02/2-Intro-Liquid%20Alkaline%20Workshop.pdf

[Brauns, Turek 2020]
"Alkaline Water Electrolysis Powered by Renewable Energy: A Review"
    https://www.mdpi.com/2227-9717/8/2/248


[Eigeldinger, Vogt 2000]
"The bubble coverage of gas evolving electrodes in a flowing electrolyte"
https://www.sciencedirect.com/science/article/pii/S0013468600005132
    -> Ref 15 of Henou 2014 for current density and theta stuff
    -> has current density equation with theta included

[Haug, Koj, Turek 2017]
"Influence of process conditions on gas purity in alkaline water electrolysis"
by Phillip Haug, Motthias Koj, Thomas Turek [2017]
https://www.sciencedirect.com/science/article/pii/S0360319916336588

[Niroula, Chaudhary, Subedi, Thapa 2003]
"Parametric Modelling and Optimization of Alkaline Electrolyzer for the Production
of Green Hydrogen" by S. Niroula, C Chaudhary, A Subedi, and B S Thapa
[2003] doi:10.1088/1757-899X/1279/1/012005
https://iopscience.iop.org/article/10.1088/1757-899X/1279/1/012005/pdf

[Vogt,Balzer 2005]
"The bubble coverage of gas-evolving elecrodes in stagnant electrolytes"
by H. Vogt and R.J. Balzer
Volume 50, Issue 10, 15 March 2005, Pages 2073-2079
https://www.sciencedirect.com/science/article/pii/S001346860400948X?via%3Dihub



"""


def ael_electrolyzer_model(X, a, b, c, d, e, f):
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


@define
class AlkalineCell(FromDictMixin):
    # Cell parameters #
    ####################
    model: str

    electrode: dict
    electrolyte: dict
    membrane: dict

    pressure_operating: float
    turndown_ratio: float
    max_current_density: float

    cell_area: float = field(init=False)

    # Electrode parameters #
    ####################
    A_electrode: float = field(init=False)  # [cm^2]
    e_a: float = field(init=False)  # [cm] anode thickness
    e_c: float = field(init=False)  # [cm] cathode thickness
    d_am: float = field(init=False)  # [cm] Anode-membrane gap
    d_cm: float = field(init=False)  # [cm] Cathode-membrane gap
    d_ac: float = field(init=False)  # [cm] Anode-Cathode gap

    # Electrolyte parameters #
    ####################
    w_koh: float = field(init=False)
    electrolyte_concentration_percent: float = field(init=False)
    M_KOH: float = field(init=False)
    M_H2O: float = field(init=False)
    m: float = field(init=False)
    M: float = field(init=False)

    # Membrane parameters #
    ####################
    A_membrane: float = field(init=False)  # [cm^2]
    e_m: float = field(init=False)  # [cm] membrane thickness

    # THIS ONE IS PRIMARLY BASED ON
    # VALUES FROM [Henou, Agbossou, 2014]
    # AND EQUATIONS FROM [Gambou, Guilbert,et al 2022]

    # num hydrogen molecules transferred per reaction
    z: int = 2

    M_H: float = 1.00784  # molecular weight of Hydrogen [g/mol]
    M_0: float = 15.999  # molecular weight of Oxygen [g/mol]
    M_K: float = 39.0983  # molecular weight of Potassium [g/mol]
    lhv: float = 33.33  # lower heating value of H2 [kWh/kg]
    hhv: float = 39.41  # higher heating value of H2 [kWh/kg]

    def __attrs_post_init__(self) -> None:
        # Cell parameters #
        ###################
        self.cell_area = self.electrode["A_electrode"]

        # Electrode parameters #
        ########################
        self.A_electrode = self.electrode["A_electrode"]
        self.e_a = self.electrode["e_a"]
        self.e_c = self.electrode["e_c"]
        self.d_am = self.electrode["d_am"]
        self.d_cm = self.electrode["d_cm"]
        self.d_ac = self.electrode["d_ac"]

        # Electrolyte parameters #
        ##########################
        self.w_koh = self.electrolyte["w_koh"]

        # Membrane parameters #
        #######################
        self.A_membrane = self.membrane["A_membrane"]
        self.e_m = self.membrane["e_m"]

        # calcluate molarity and molality of KOH solution
        self.electrolyte_concentration_percent = self.w_koh / 100
        self.create_electrolyte()

    def calculate_bubble_rate_coverage(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        returns:
            -theta: bubble rate coverage
            -epsilon: bulk bubble rate?
        Reference:
        [Hammoudi,Henao, 2012] Eqn 44
        [Vogt,Balzer 2005]: value for J_lim
        [Niroula, Chaudhary, Subedi, Thapa 2003]: Eqn 17
        [Gambou, Guilbert,et al 2022] Eqn 19 for Pv_H2O
        """

        # VERIFIED for j=0.01 A/cm^2 and T_C=50 degC (Figure 4 from H. Vogt)
        # NOTE: H. Vogt paper uses 3M KOH solution, but we're using closer to 6

        T_k = convert_temperature([T_C], "C", "K")[0]
        J_lim = 30  # [A/cm^2] [Vogt,Balzer 2005]
        T_amb = T_k = convert_temperature([25], "C", "K")[0]
        j = I / self.A_electrode  # [A/cm^2] "nominal current density"

        # Eqn 19 of [Gambou, Guilbert,et al 2022]
        Pv_H20 = np.exp(
            81.6179 - (7699.68 / T_k) - (10.9 * np.log(T_k)) + (T_k * (9.5891 * 1e-3))
        )

        # theta is bubble rate coverage
        # Eqn 17 of [Niroula, Chaudhary, Subedi, Thapa 2003]
        theta = (
            (self.pressure_operating / (self.pressure_operating - Pv_H20))
            * (-97.25 + 182 * (T_k / T_amb) - 84 * ((T_k / T_amb) ** 2))
            * (j / J_lim) ** (0.3)
        )
        # theta = (-97.25 + 182*(T_k/T_amb)-84*((T_k/T_amb)**2))*(j/J_lim)**(0.3)
        # ^^Eqn 44 of [Hammoudi,Henao, 2012]
        # epsilon is the covering rate bubbling?
        # (below) [Hammoudi,Henao, 2012] page 13906 in text at end of section 3.1
        epsilon = (2 / 3) * theta  # bulk bubbling
        return [theta, epsilon]

    def calc_current_density(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        returns:
            -current_dens [A/cm^2] which is equivalent to "J"
        Reference:
        [Henou, Agbossou, 2014]: Eqn 6
        """
        # actual current density reflecting impact of bubble rate coverage
        theta_epsilon = self.calculate_bubble_rate_coverage(T_C, I)
        theta = theta_epsilon[0]
        A_electrode_eff = self.A_electrode * (1 - theta)  # [cm^2]
        current_dens = I / A_electrode_eff  # [A/cm^2]
        return current_dens

    def create_electrolyte(self):
        """
        depends on the user-input self.electrolyte_concentration_percent

        sets:
            -self.m [mol/kg]:  molality of KOH solution
            -self.M [mol/L] molarity of KOH solution [mol/L]
        Reference: [none]
        """

        # https://www.sciencedirect.com/science/article/pii/S0360319916336588
        # ^states that 30 wt% KOH has a molar concentration of 6.9 mol/L
        # solute is KOH
        solution_weight_g = 1000
        self.M_KOH = self.M_0 + self.M_H + self.M_K  # [g/mol]
        grams_of_solute = solution_weight_g * (self.electrolyte_concentration_percent)
        # kg_of_solute=(1/1000)*grams_of_solute
        moles_of_solute = grams_of_solute / self.M_KOH  # [mols of solute / solution]

        # solvent is water
        self.M_H2O = 2 * self.M_H + self.M_0  # [g/mol]
        grams_of_solvent = solution_weight_g * (
            1 - self.electrolyte_concentration_percent
        )
        # moles_of_solvent = grams_of_solvent/self.M_H20 #[mol of solvent / solution]
        kg_of_solvent = (1 / 1000) * grams_of_solvent

        density_of_water = 1  # [g/mL] #TODO: could be temperature dependent
        density_of_KOH = 2.12  # [g/mL] #TODO: could be temperature dependent
        volume_of_water = grams_of_solvent / density_of_water  # mL
        volume_of_KOH = grams_of_solute / density_of_KOH  # mL
        volume_of_solution = (volume_of_water + volume_of_KOH) / 1000  # L

        # molality = mol of solute / kg of solvent
        molality = moles_of_solute / kg_of_solvent  # mol/kg
        # molarity is mol/kg = mol of solute / L of solution
        molarity = moles_of_solute / volume_of_solution  # mol/L
        # % solution = amount of solute / amount of solution

        self.m = molality  # NOTE: THIS HAS BEEN VALIDATED
        self.M = molarity  # NOTE: THIS HAS BEEN VALIDATED

    def gas_purity_and_flow(self):
        # [Haug, Koj, Turek 2017]

        # TODO: add capability to model
        # cell_vol = 0.7  # L
        # avg_flow_rate = 0.33 * (1 / 60)  # L/min -> L/sec (40% pump rate)
        # cathode_flow_rate = 0.375 * (1 / 60)  # [L/sec] Table 5
        # anode_flow_rate = 0.285 * (1 / 60)  # [L/sec] Table 5

        # lower_explosion_limit = 3.8  # mol% H2/O2
        # upper_explosion_limit = 95.4  # mol% H2?
        # safety_lim_H2_in_O2 = lower_explosion_limit / 2  # 2% h2 in O2
        # shutdown limit is 2% H2 in O2

        pass

    def calc_cell_voltage(self, I, T_C):
        """
        I [A]: current
        T_C [C]: temperature
        return :: V_cell [V/cell]: total cell voltage

        Reference: [Gambou, Guilbert,et al 2022]: Eqn 4
        """
        V_rev = self.calc_Urev(T_C, self.pressure_operating)
        V_act_a, V_act_c = self.calc_activation_overpotential(T_C, I)
        V_ohm = self.calc_ohmic_overpotential(T_C, I)
        V_cell = V_rev + V_ohm + V_act_a + V_act_c  # Eqn 4
        return V_cell

    def calc_Urev(self, T_C, P):
        """
        P [bar]: operating pressure
        T_C [C]: temperature
        returns:: Urev [V/cell]:

        Reference:
        [Gambou, Guilbert,et al 2022]: Eqn 14,17-20
        """

        # P = 1 #gas pressure [bar], TODO: double check!
        T_K = convert_temperature([T_C], "C", "K")[0]

        # Eqn 17
        a = (
            (-0.0151 * self.m)
            - (1.6788 * (1e-3) * (self.m**2))
            + (2.2588 * (1e-5) * (self.m**3))
        )

        # Eqn 18 #[bar]
        b = (
            1
            - ((1.26062 * (1e-3)) * self.m)
            + ((5.6024 * 1e-4) * (self.m**2))
            - ((self.m**3) * (7.8228 * 1e-6))
        )
        # Pv_H20 is vapor pressure of pure water [bar]
        # Eqn 19
        Pv_H20 = np.exp(
            81.6179 - (7699.68 / T_K) - (10.9 * np.log(T_K)) + (T_K * (9.5891 * 1e-3))
        )

        # alpha_h20: water activity of electrolyte solution based on molality [mol/kg]
        # valid for molality ranging from 2-18 mol/kg
        # Eqn 20
        alpha_h20 = np.exp(
            (-0.05192 * self.m)
            + (0.003302 * (self.m**2))
            + (((3.177 * self.m) - (2.131 * self.m**2)) / T_K)
        )

        # Pv_KOH: [bar] vapor pressure of KOH solution
        # Eqn 16
        Pv_KOH = np.exp((2.302 * a) + (b * np.log(Pv_H20)))

        Urev0 = self.calc_open_circuit_voltage(T_C)

        # Eqn 14
        U_rev = Urev0 + ((R * T_K) / (self.z * F)) * np.log(
            ((P - Pv_KOH) ** 1.5) / alpha_h20
        )

        return U_rev

    def calc_activation_overpotential(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        returns::
            V_act_a [V/cell]: resistance of nickle anode
            V_act_a [V/cell]: resistance of nickle cathode
        Reference: [Niroula, Chaudhary, Subedi, Thapa 2003] Eqn 11-16
        """

        # validated against Figure 5 of Reference
        j_eff = self.calc_current_density(T_C, I)
        ja = j_eff  # [A/cm^2]
        jc = j_eff  # [A/cm^2]
        T_anode = convert_temperature([T_C], "C", "K")[0]
        T_cathode = convert_temperature([T_C], "C", "K")[0]
        # Eqn 14 anode charge transfer coeff
        alpha_a = 0.0675 + 0.00095 * T_anode
        # Eqn 15 cathode charge transfer coeff
        alpha_c = 0.1175 + 0.00095 * T_cathode

        # Table 1
        delta_Ga = 41500  # [J/mol*K]
        delta_Gc = 23450  # [J/mol*K]
        jref_0a = 1.34535 * 10 ** (-5)  # [A/cm^2]
        jref_0c = 1.8456 * 10 ** (-3)  # [A/cm^2]
        Tref = convert_temperature([25], "C", "K")[0]
        gamma_a = 1.25  # anode roughness factor
        gamma_c = 1.05  # cathode roughness factor
        # Eqn 16
        j0c = (
            gamma_c
            * jref_0c
            * np.exp((-1 * delta_Gc / R) * ((1 / T_cathode) - (1 / Tref)))
        )
        # Eqn 16
        j0a = (
            gamma_a
            * jref_0a
            * np.exp((-1 * delta_Ga / R) * ((1 / T_anode) - (1 / Tref)))
        )
        # Eqn 13 - Tafel slope for anode
        ba = (R * T_anode) / (self.z * F * alpha_a)
        # Eqn 13 - Tafel slope for cathode
        bc = (R * T_anode) / (self.z * F * alpha_c)
        # Eqn 11 - anode activation energy
        V_act_a = ba * np.maximum(0, np.log(ja / j0a))
        # Eqn 12 - cathode activation energy
        V_act_c = bc * np.maximum(0, np.log(jc / j0c))

        return V_act_a, V_act_c

    def calc_electrode_resistance(self, T_C):
        """
        I [A]: current
        T_C [C]: temperature
        returns::
            Ra [Ohms]: resistance of nickle anode
            Rc [Ohms]: resistance of nickle cathode
        Reference: [Niroula, Chaudhary, Subedi, Thapa 2003]: Eqn 20-21, Table 1
        """
        # nickle anode and cathode
        # Table 1
        tref = 25
        temp_coeff = 0.00586  # 1/degC
        # resistivity of 100% dense electrode at tref
        rho_nickle_0 = 6.4 * 10 ** (-6)  # [Ohm*cm]
        # porosity of electrode
        epsilon_Ni = 0.3
        # Eqn 21 - effective resistance of electrode
        rho_nickle_eff = rho_nickle_0 / ((1 - epsilon_Ni) ** 1.5)
        # Eqn 20 - resistivity of anode
        Ra = (
            rho_nickle_eff
            * (self.e_a / self.A_electrode)
            * (1 + (temp_coeff * (T_C - tref)))
        )
        # Eqn 20 - resistivity of cathode
        Rc = (
            rho_nickle_eff
            * (self.e_c / self.A_electrode)
            * (1 + (temp_coeff * (T_C - tref)))
        )

        # [Gambou, Guilbert,et al 2022]: Electrode resistances [Eqn 29-30]
        # [Henou, Agbossou, 2014]: Equations 13
        # Also - one of these used La and Lc instead of e_a and e_c
        # but those were described as the "height" of the electrode ...
        # ^ these gave a negative value for the conductivity of nickle...

        return Ra, Rc  # Ohms

    def calc_electrolyte_resistance(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        returns::
            -R_ele_bf [Ohms]: KOH electrolyte resistance (not due to bubbles)
            -R_ele_b [Ohms]: KOH electrolyte resistance due to bubbles
        Reference:
            [Gambou, Guilbert,et al 2022]: Eqn 31-35
            [Brauns,2021]: Eqn 2
            [Henou, Agbossou, 2014] Eqn 19,20
        """

        T_K = convert_temperature([T_C], "C", "K")[0]

        # M is electrolyte molarity concentration mol/L
        # sigma_bf is bubble-free electrolyte conductivity
        # ->[Eqn 33] of [Gambou, Guilbert,et al 2022]
        #   sigma_bf is in units S/m - but units aren't converted
        #
        sigma_bf = (
            -204.1 * self.M
            - 0.28 * self.M**2
            + 0.5332 * self.M * T_K
            + 20720 * (self.M / T_K)
            + 0.1043 * self.M**3
            - 0.00003 * (self.M**2 * T_K**2)
        )
        # sigma_bf verified
        # sigma_bf is the same as sigma_KOH -> could probably replace with:
        # sigma_bf = self.calculate_KOH_conductivity(T_C)

        # R_ele_bf: Bubble-free electrolyte resistance
        # Eqn 32 of [Gambou, Guilbert,et al 2022] and Eqn 19 of [Henou, Agbossou, 2014]

        R_ele_bf = (100 / sigma_bf) * (
            (self.d_am / self.A_electrode) + (self.d_cm / self.A_electrode)
        )
        # Eqn 2 of [Brauns,2021] says R_eg=(1/(sigma_koh))*((dcs+das)/A_el)
        # where A_el is the metal area, not the entire area (which has holes)
        # Eqn 34 of [Gambou, Guilbert,et al 2022] and Eqn 20 of [Henou, Agbossou, 2014]
        # Resistance due to bubbles
        theta_epsilon = self.calculate_bubble_rate_coverage(T_C, I)
        epsilon = theta_epsilon[1]
        # R_ele_b=R_ele_bf*((1/(1-epsilon)**(3/2))-1)
        # R_ele_b: Bubble resistance
        R_ele_b = R_ele_bf * ((1 / ((1 - epsilon) ** (3 / 2))) - 1)
        # ^Bruggman equation
        # Eqn 31 of [Gambou, Guilbert,et al 2022] (below)
        # Rele=R_ele_bf + R_ele_b #Ohms?

        return R_ele_bf, R_ele_b  # Ohms

    def calc_membrane_resistance(self, T_C):
        """
        T_C [C]: temperature
        returns:: R_mem [Ohms]: resistance of Zirfon membrane

        Reference:
        [Gambou, Guilbert,et al 2022]: Eqn 36
        [Henou, Agbossou, 2014]: Eqn 21
        [NEL]: TODO add slide
        """

        # NOTE: THIS HAS BEEN VERIFIED
        # for T_C=80, S_mem=54.48 and outputs Rmem=0.23 ohm*cm^2
        # which is similar to NEL which states that it R_mem=0.25 ohm*cm^2
        # [Gambou, Guilbert,et al 2022]
        # S_mem=54.48 # membrane surface area in cm^2

        Rmem = (0.06 + 80 * np.exp(T_C / 50)) / (
            10000 * self.A_membrane
        )  # Equation 36 - Ohms
        # ^ Equation 21 of [Henou, Agbossou, 2014]
        # output: Rmem=0.23 ohm*cm^2
        # ^^Electrolyzer membrane resistance made of Zirfon material of 0.5mm thickness

        # NOTE: below is alternative method to calculate it
        # -> from: [TODO add eqn] of [Niroula, Chaudhary, Subedi, Thapa 2003]
        # sigma_bf = (-204.1*self.M -0.28*self.M**2 + 0.5332*self.M*T_K +
        # 20720*(self.M/T_K) + 0.1043*self.M**3 - 0.00003*(self.M**2 * T_K**2))
        # tau_mem = 2.18 #or 3.14
        # epsilon_mem = 0.42 #porosity
        # omega_mem = 0.85 #wettability factor

        # R_mem = (1/sigma_bf)*((tau_mem**2)*self.e_m)/
        # (omega_mem*epsilon_mem*self.A_membrane)

        # Rm = (1/sigma_bf)*((tau_m**2)*d_mem)*(omega_m*epsilon_m*A_membrane)
        # which gives 0.3366, but not temperature dependent?
        return Rmem  # ohms?

    def calc_total_resistance(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        returns :: R_tot [Ohms]:

        Reference: [Gambou, Guilbert,et al 2022]: Eqn 27
        """

        R_a, R_c = self.calc_electrode_resistance(T_C)
        R_electrode = R_a + R_c
        R_ele_bf, R_ele_b = self.calc_electrolyte_resistance(T_C, I)  # [Ohms]
        R_electrolyte = R_ele_bf + R_ele_b
        R_membrane = self.calc_membrane_resistance(T_C)  # [Ohms] VERIFIED for Ohm*cm^2
        R_tot = R_electrode + R_electrolyte + R_membrane  # Ohm

        return R_tot

    def calc_ohmic_overpotential(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        return :: V_ohm [V/cell]: overpotential due to resistive losses

        Reference: [Gambou, Guilbert,et al 2022] Eqn 10
            -> uses current instead of current density
        """

        R_tot = self.calc_total_resistance(T_C, I)  # Ohms
        # NOTE: I'm really not sure what units to use anymore...
        V_ohm = I * R_tot  # [V/cell]
        return V_ohm

    def calc_open_circuit_voltage(self, T_C):
        """
        I [A]: current
        T_C [C]: temperature
        return :: E_rev0 [V/cell]: open-circuit voltage

        TODO: Are we correcting for temperature twice? U_rev0 should be just 1.229 and
        never change (possibly?)

        Reference: [Gambou, Guilbert,et al 2022]: Eqn 14
        """
        # General Nerst Equation
        # Eqn 14 of [Gambou, Guilbert,et al 2022]
        T_K = convert_temperature([T_C], "C", "K")[0]
        E_rev0 = (
            1.5184
            - (1.5421 * (10 ** (-3)) * T_K)
            + (9.523 * (10 ** (-5)) * T_K * np.log(T_K))
            + (9.84 * (10 ** (-8)) * (T_K**2))
        )
        # OR should this just be 1.229?
        # E_rev_fake = 1.229
        return E_rev0

    def calc_faradaic_efficiency(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        return :: eta_F [-]: Faraday's efficiency
        Reference: [Oystein Ulleberg, 2003] Eqn 9, Table 3
        """
        # f1 and f2 values from Table 3
        f1 = 250  # [mA^2/cm^4]
        f2 = 0.96  # [-]
        j = self.calc_current_density(T_C, I)  # [A/cm^2]
        j *= 1000  # [mA/cm^2]

        # TODO: make coefficients based on temperature
        # HYSOLAR is first 3 then last one is PHOEBUS
        # Table 3 of [Oystein Ulleberg, 2003]
        #     T_opt=[40,60,80,80] #deg C
        #     f_1=[150,200,250,250] #mA^2/cm^4
        #     f_2=[0.99,0.985,0.98,0.96] #[0-1]

        # Eqn 9 from [Oystein Ulleberg, 2003]
        eta_F = f2 * (j**2) / (f1 + j**2)
        return eta_F

    def calc_mass_flow_rate(self, T_C, I):
        """
        I [A]: current
        T_C [C]: temperature
        return :: mfr [kg/s]: mass flow rate of H2
        Reference: [Oystein Ulleberg, 2003]: Eqn 10
        """

        eta_F = self.calc_faradaic_efficiency(T_C, I)
        # Eqn 10 [mol/sec]
        h2_prod_mol = eta_F * I / (self.z * F)
        mfr = self.M_H * self.z * h2_prod_mol  # [g/sec]
        # z is valency number of electrons transferred per ion
        # for oxygen, z=4
        mfr = mfr / 1e3  # [kg/sec]
        return mfr
        # h2_prod is in mol/s
