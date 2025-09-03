"""This module defines a Hydrogen Electrolyzer Stack."""

from typing import Union

import numpy as np
import scipy
import pandas as pd
import rainflow
from attrs import field, define
from scipy.signal import tf2ss, cont2discrete
from scipy.constants import physical_constants

from electrolyzer.tools.type_dec import NDArrayFloat, FromDictMixin, array_converter
from electrolyzer.tools.validators import contains
from electrolyzer.simulation.cell_models.pem import PEMCell, PEM_electrolyzer_model
from electrolyzer.simulation.cell_models.alkaline import (
    AlkalineCell,
    ael_electrolyzer_model,
)


F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]


@define
class Stack(FromDictMixin):
    # Stack parameters #
    ####################
    dt: float = field()
    cell_type: str = field(validator=contains(["PEM", "alkaline"]))
    temperature: float = field()
    n_cells: int = field()

    degradation: dict = field()
    cell_params: dict = field()

    stack_rating_kW: float = field(default=None)
    include_degradation_penalty: bool = field(default=True)
    # If degradation results in hydrogen losses, hydrogen_degradation_penalty=True
    # If degradation results in more power consumed, hydrogen_degradation_penalty=False
    hydrogen_degradation_penalty: bool = field(default=True)
    eol_eff_percent_loss: float = field(init=False)
    d_eol: float = field(init=False)

    max_current: float = field(default=1000)  # TODO this is a bad default, fix later

    min_power: float = field(default=None)

    turndown_ratio: float = field(init=False)
    cell_area: float = field(init=False)

    # initialized in __attrs_post_init
    cell: Union[PEMCell, AlkalineCell] = field(init=False)
    fit_params: NDArrayFloat = field(init=False)
    stack_rating: float = field(init=False)
    electrolyzer_model = field(init=False)

    # Degradation state #
    #####################

    rate_steady: float = field(init=False)  # conversion factor for steady degradation
    rate_fatigue: float = field(init=False)  # conversion factor for fatigue degradation
    rate_onoff: float = field(init=False)  # conversion factor for on off degradation

    # [s] amount of time this electrolyzer stack has been running
    uptime: float = field(init=False, default=0)

    cell_voltage: float = field(init=False, default=0)

    # [V] degradation penalty from steady operation only
    d_s: float = field(init=False, default=0)

    # fatigue value for tracking fatigue in terms of "stress cycles"
    # rainflow counting
    rf_track: float = field(init=False, default=0)

    # [V] running count of fatigue voltage penalty
    fatigue_history: float = field(init=False, default=0)

    hourly_counter: float = field(init=False, default=0)
    hour_change: bool = field(init=False, default=False)
    voltage_signal: NDArrayFloat = field(
        init=False, default=[], converter=array_converter
    )
    voltage_history: NDArrayFloat = field(
        init=False, default=[], converter=array_converter
    )
    degradation_history: NDArrayFloat = field(
        init=False, default=[], converter=array_converter
    )
    power_input_history: NDArrayFloat = field(
        init=False, default=[], converter=array_converter
    )

    # [V] degradation from fluctuating power only
    d_f: float = field(init=False, default=0)

    # numer of times the stack has been turned off
    cycle_count: int = field(init=False, default=0)

    # [V] degradation from on/off cycling only
    d_o: float = field(init=False, default=0)

    # [V] running degradation voltage penalty
    V_degradation: float = field(init=False, default=0)

    # Stack dynamics #
    ##################

    # Current (A)
    I: float = field(init=False, default=0.0)

    # 10 minute startup procedure
    stack_on: bool = field(init=False, default=False)
    stack_waiting: bool = field(init=False, default=False)

    # [s] 10 minute base turn on delay, for large time steps
    base_turn_on_delay: float = 600

    # [s] 10 minute time delay for PEM electrolyzer startup procedure
    # (set in __attrs_post_init__)
    turn_on_delay: float = field(init=False)

    # keep track of when the stack was last turned on
    turn_on_time: float = field(init=False)

    # keep track of when the stack was last turned off
    turn_off_time: float = field(init=False)

    # wait time for partial startup procedure (set in __attrs_post_init)
    wait_time: float = field(init=False)

    # [s] total time of simulation
    time: float = field(init=False, default=0)

    # [s] time constant
    # https://www.sciencedirect.com/science/article/pii/S0360319911021380
    # section 3.4 # noqa
    tau: float = 5

    stack_state: float = field(init=False, default=0)

    # state space, (set in __attrs_post_init)
    DTSS: NDArrayFloat = field(init=False)

    # whether 1st order dynamics should be ignored according to dt size
    ignore_dynamics: bool = field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        # Stack parameters #
        ####################
        self.eol_eff_percent_loss = self.degradation["eol_eff_percent_loss"]
        if self.cell_type == "PEM":
            # initialize electrolzyer cell model
            self.cell = PEMCell.from_dict(self.cell_params["PEM_params"])

            # set degradation rates
            self.rate_steady = self.degradation["PEM_params"]["rate_steady"]
            self.rate_fatigue = self.degradation["PEM_params"]["rate_fatigue"]
            self.rate_onoff = self.degradation["PEM_params"]["rate_onoff"]

            # electrolyzer_model for current calculation
            self.electrolyzer_model = PEM_electrolyzer_model

        elif self.cell_type == "alkaline":
            # initialize electrolyzer cell model
            self.cell = AlkalineCell.from_dict(self.cell_params["ALK_params"])

            # set degradation rates
            self.rate_steady = self.degradation["ALK_params"]["rate_steady"]
            self.rate_fatigue = self.degradation["ALK_params"]["rate_fatigue"]
            self.rate_onoff = self.degradation["ALK_params"]["rate_onoff"]

            # electrolyzer_model for current calculation
            self.electrolyzer_model = ael_electrolyzer_model

        # [kW] nameplate power rating
        self.stack_rating_kW = self.stack_rating_kW or self.calc_stack_power(
            self.max_current
        )

        self.stack_rating = self.stack_rating_kW * 1e3  # [W] nameplate rating

        # set minimum power
        if self.cell_type == "PEM":
            self.turndown_ratio = self.cell_params["PEM_params"]["turndown_ratio"]
        elif self.cell_type == "alkaline":
            self.turndown_ratio = self.cell_params["ALK_params"]["turndown_ratio"]
        self.min_power = self.min_power or (self.turndown_ratio * self.stack_rating)

        self.fit_params = self.create_polarization()

        # Stack dynamics #
        ##################

        # If the time step is bigger than the 1st order time constant, ignore dynamics
        if self.dt > self.tau:
            self.ignore_dynamics = True

        # Remove turn on delay for large time steps
        if self.dt > 2 * self.base_turn_on_delay:
            self.turn_on_delay = 0
        else:
            self.turn_on_delay = self.base_turn_on_delay

        self.turn_on_time = 0
        self.turn_off_time = -self.turn_on_delay

        self.wait_time = np.min(
            [
                (self.turn_on_time - self.turn_off_time),
                self.turn_on_delay,
            ]
        )

        self.DTSS = self.calc_state_space()
        self.d_eol = self.calc_end_of_life_voltage()

    def run_power_deg_penalty(self, P_in):
        """Run the stack with a degradation penalty applied to the power consumption.
        Power consumed may be greater than power input.

        Args:
            P_in (float): stack power input in Wdc

        Returns:
            2-element tuple containing

            - **I_stack** (float): stack current in Amps
            - **V_cell** (float): cell voltage in Volts
        """

        I_stack = self.electrolyzer_model(
            (P_in / 1e3, self.temperature), *self.fit_params
        )
        V_cell = self.cell.calc_cell_voltage(I_stack, self.temperature)

        return I_stack, V_cell

    def run_h2_deg_penalty(self, P_in):
        """Run the stack with a degradation penalty applied to the hydrogen production.
        Power consumed should be nearly equal to power input.

        Args:
            P_in (float): stack power input in Wdc

        Returns:
            2-element tuple containing

            - **I_stack** (float): stack current in Amps
            - **V_cell** (float): cell voltage in Volts
        """

        I_nom = self.electrolyzer_model(
            (P_in / 1e3, self.temperature), *self.fit_params
        )
        V_cell = self.cell.calc_cell_voltage(I_nom, self.temperature)
        eff_mult = np.nan_to_num((V_cell + self.V_degradation) / V_cell)
        I_stack = np.nan_to_num(I_nom / eff_mult)

        return I_stack, V_cell

    def run(self, P_in):
        """Run the stack for smoe input power.

        Args:
            P_in (float): stack power input in Wdc

        Returns:
            3-element tuple containing

            - **H2_mfr** (float): hydrogen mass flow rate in kg/sec
            - **H2_mass_out** (float): hydrogen mass produced in kg/dt
            - **power_left** (float): remaining power, P_in and power consumed

        """
        self.update_status()
        if self.hydrogen_degradation_penalty:
            I_stack, V_cell = self.run_h2_deg_penalty(P_in)
        else:
            I_stack, V_cell = self.run_power_deg_penalty(P_in)

        H2_mfr, H2_mass_out, power_left = self.run_stack(I_stack, V_cell, P_in)
        return H2_mfr, H2_mass_out, power_left

    def run_stack(self, I_stack, V_cell, P_in):
        """Run the stack for a given stack current, cell voltage, and power input.
        Updates the cell degradation, updates dynamics, calculates hydrogen mass
        flow rate and production, and remaining power.

        Args:
            I_stack (float): stack current input in Amps
            V_cell (float): cell voltage in Volts
            P_in (float): stack power input in Wdc
        Returns:
            3-element tuple containing

            - **H2_mfr** (float): hydrogen mass flow rate in kg/sec
            - **H2_mass_out** (float): hydrogen mass produced in kg/dt
            - **power_left** (float): difference in P_in and power consumed in Watts
        """

        if self.stack_on:
            power_left = P_in

            self.I = I_stack

            if self.include_degradation_penalty:
                V_cell += self.V_degradation

            self.update_temperature(I_stack, V_cell)
            self.update_degradation()
            power_left -= self.calc_stack_power(I_stack, V_cell) * 1e3
            H2_mfr = (
                self.cell.calc_mass_flow_rate(self.temperature, I_stack) * self.n_cells
            )
            self.stack_state, H2_mfr = self.update_dynamics(H2_mfr, self.stack_state)

            H2_mass_out = H2_mfr * self.dt
            self.uptime += self.dt

        else:
            H2_mfr = 0
            H2_mass_out = 0
            self.stack_state, H2_mfr = self.update_dynamics(H2_mfr, self.stack_state)

            if self.stack_waiting:
                self.uptime += self.dt
                self.I = I_stack
                self.update_temperature(I_stack, V_cell)
                self.update_degradation()
                power_left = 0
            else:
                power_left = P_in
                V_cell = 0

        self.cell_voltage = V_cell
        self.voltage_history = np.append(self.voltage_history, [V_cell])
        self.degradation_history = np.append(
            self.degradation_history, [self.V_degradation]
        )
        self.power_input_history = np.append(self.power_input_history, [P_in])
        # check if it is an hour to decide whether to calculate fatigue
        hourly_temp = self.hourly_counter
        self.time += self.dt
        self.hourly_counter = self.time // 3600
        if hourly_temp != self.hourly_counter:
            self.hour_change = True
            self.voltage_signal = self.voltage_history
            self.voltage_history = np.array([])
        else:
            self.hour_change = False

        return H2_mfr, H2_mass_out, power_left

    # ------------------------------------------------------------
    # Polarization model
    # ------------------------------------------------------------
    def create_polarization(self):
        interval = 10.0
        currents = np.arange(0, self.max_current + interval, interval)
        pieces = []
        prev_temp = self.temperature
        for temp in np.arange(40, 60 + 5, 5):
            # for temp in np.arange(self.temperature - 5, self.temperature + 10, 5):
            self.temperature = temp
            powers = self.calc_stack_power(currents)
            tmp = pd.DataFrame({"current_A": currents, "power_kW": powers})
            tmp["temp_C"] = temp
            pieces.append(tmp)
        self.temperature = prev_temp
        df = pd.concat(pieces)

        # assign initial values and solve a model
        paramsinitial = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        # use curve_fit routine
        fitobj, fitcov = scipy.optimize.curve_fit(
            self.electrolyzer_model,
            (df.power_kW.values, df.temp_C.values),
            df.current_A.values,
            p0=paramsinitial,
        )

        return fitobj

    def convert_power_to_current(self, Pdc, T):
        """Estimate stack current for a given power operating point and temperature.

        Args:
            Pdc (float): stack power in kWdc
            T (float): stack temperature in Celsius

        Returns:
            float: ``Idc`` stack current in Amps
        """
        Idc = self.electrolyzer_model((Pdc, T), *self.fit_params)
        return Idc

    def curtail_power(self, P_in):
        """Curtail power if power exceeds stack rating

        Args:
            P_in (float): input power in kWdc

        Returns:
            float | array: input power in kWdc saturated at stack rated power.
        """
        return np.where(P_in > self.stack_rating_kW, self.stack_rating_kW, P_in)

    def calc_fatigue_degradation(self, voltage_signal):
        """
        Args:
            voltage_signal (float | array): the voltage signal from the last
            3600 seconds
        Returns:
            float | array: ``voltage_penalty`` the degradation penalty from
            variable operation in Volts
        """
        # based off degradation due to square waves of different frequencies
        # from results in https://iopscience.iop.org/article/10.1149/2.0231915jes

        # nonzero voltage signal so that it does not double count power cycling
        voltage_signal = voltage_signal[np.nonzero(voltage_signal)]

        # rainflow counting for fatigue
        rf_cycles = rainflow.count_cycles(voltage_signal, nbins=10)
        rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
        self.rf_track += rf_sum  # running sum of the fatigue value

        return self.rate_fatigue * self.rf_track

    def calc_steady_degradation(self):
        # based off degradation due to steady operation
        # from results in https://iopscience.iop.org/article/10.1149/2.0231915jes

        d_s = self.d_s + self.rate_steady * self.cell_voltage * self.dt

        self.d_s = d_s
        return d_s

    def calc_onoff_degradation(self):
        # degradation due to shut downs based off the results in
        # https://iopscience.iop.org/article/10.1149/2.0421908jes/meta

        d_o = self.rate_onoff * self.cycle_count
        self.d_o = d_o
        return d_o

    def update_degradation(self):
        if self.hour_change:  # only calculate fatigue degradation every hour
            # fatigue only counts the nonzero voltage fluctuations since transition to
            # and from V = 0 are captured with on/off cycles.
            voltage_signal_nz = self.voltage_signal[np.nonzero(self.voltage_signal)]

            # to avoid a divide by zero, only proceed if there are nonzero values in the
            # voltage signal.
            if len(voltage_signal_nz) > 0:
                voltage_perc = (max(voltage_signal_nz) - min(voltage_signal_nz)) / max(
                    voltage_signal_nz
                )

                # Only penalize if more than 5% difference in voltage
                if voltage_perc > 0.05:
                    self.fatigue_history = self.calc_fatigue_degradation(
                        self.voltage_signal
                    )

        self.d_f = self.fatigue_history

        self.V_degradation = (
            self.calc_steady_degradation()
            + self.calc_onoff_degradation()
            + self.fatigue_history
        )

    def update_temperature(self, I, V):
        # placeholder
        return self.temperature

    def update_dynamics(self, H2_mfr_ss, stack_state):
        """This is really just a filter on the steady state mfr from
        time step to time step.

        Args:
            H2_mfr_ss (float): steady state mass flow rate
            stack_state (float): previous mfr state

        Returns:
            2-element tuple containing

            - **next_state** (float): next mfr state
            - **H2_mfr_actual** (float): actual mfr according to the filter
        """

        if self.ignore_dynamics:
            H2_mfr_actual = H2_mfr_ss
            next_state = self.stack_state
        else:
            x_k = stack_state
            x_kp1 = self.DTSS[0] * x_k + self.DTSS[1] * H2_mfr_ss
            y_kp1 = self.DTSS[2] * x_k + self.DTSS[3] * H2_mfr_ss
            next_state = x_kp1[0][0]
            H2_mfr_actual = y_kp1[0][0]

        return next_state, H2_mfr_actual

    def calc_state_space(self):
        """
        Initialize the state space matrices
        """
        tau = self.tau
        dt = self.dt
        num = [1]
        den = [tau, 1]
        ss_c = tf2ss(num, den)
        ss_d = cont2discrete((ss_c[0], ss_c[1], ss_c[2], ss_c[3]), dt, "zoh")
        return [ss_d[0], ss_d[1], ss_d[2], ss_d[3]]

    def update_status(self):
        """Update the stack status if the stack is waiting. If the stack is waiting
        and has waited long enough to be on, this method updates the stack status to on.
        """
        # Change the stack to be truly on if it has waited long enough
        if self.stack_on:
            return

        if self.stack_waiting:
            if (self.turn_on_time + self.wait_time) <= self.time:
                self.stack_waiting = False
                self.stack_on = True

    def turn_stack_off(self):
        """Turn the stack off if the stack is on or watiting.
        Updates the cycle count, waiting period, turn off time, and stack status.
        """
        if self.stack_on or self.stack_waiting:
            # record turn off time to adjust waiting period
            self.turn_off_time = self.time
            self.stack_on = False
            self.stack_waiting = False
            self.cycle_count += 1

            # adjust waiting period
            self.wait_time = np.max(
                [0, self.wait_time - (self.turn_off_time - self.turn_on_time)]
            )

    def turn_stack_on(self):
        """Turn the stack on if the stack is off or watiting.
        Updates the waiting period, turn on time, and stack status.
        """
        if self.stack_on:
            return

        if not self.stack_waiting:
            self.turn_on_time = self.time

        # record turn on time to adjust waiting period
        self.stack_waiting = True

        # adjust waiting period
        self.wait_time = np.min(
            [
                self.wait_time + (self.turn_on_time - self.turn_off_time),
                self.turn_on_delay,
            ]
        )

    def calc_stack_power(self, Idc, V=None):
        """
        Args:
            Idc (float): stack current in Amps
            V (float, optional): stack voltage

        Returns:
            float: ``Pdc`` [kW] stack power
        """
        V = V or (self.cell.calc_cell_voltage(Idc, self.temperature))
        Pdc = Idc * V * self.n_cells
        Pdc = Pdc / 1000.0  # [kW]

        return Pdc

    def calc_electrolysis_efficiency(self, Pdc, mfr):
        """Calculate the efficiency of the stack in kWh/kg, %-HHV and %-LHV.

        Args:
            Pdc (float): stack power in kW
            mfr (float): mass flow rate of hydrogen in kg/hr

        Returns:
            3-element tuple containing

            - **eta_kWh_per_kg**: efficiency in kWh/kg
            - **eta_hhv_percent**: efficiency as %-HHV
            - **eta_lhv_percent**: efficiency as %-LHV
        """
        eta_kWh_per_kg = Pdc / mfr
        eta_hhv_percent = self.cell.hhv / eta_kWh_per_kg * 100.0
        eta_lhv_percent = self.cell.lhv / eta_kWh_per_kg * 100.0

        return (eta_kWh_per_kg, eta_hhv_percent, eta_lhv_percent)

    def calc_end_of_life_voltage(self):
        """Calculate the end-of-life cell degradation voltage based on the
        ``eol_eff_percent_loss`` parameter.

        Returns:
            d_eol (float): cell degradation in Volts that indicates end-of-life.
        """

        # efficiency drop that indicates end-of-life as a percentage
        eol_eff_mult = (100 + self.eol_eff_percent_loss) / 100
        V_cell_bol = self.cell.calc_cell_voltage(self.max_current, self.temperature)
        H2_mfr_bol = (
            self.cell.calc_mass_flow_rate(self.temperature, self.max_current)
            * self.n_cells
        )

        H2_mfr_eol = H2_mfr_bol / eol_eff_mult
        i_eol_no_faradaic_loss = (H2_mfr_eol * 1e3 * self.cell.n * F) / (
            1 * self.n_cells * self.cell.M * self.dt
        )
        n_f = self.cell.calc_faradaic_efficiency(
            self.temperature, i_eol_no_faradaic_loss
        )
        i_eol = (H2_mfr_eol * 1e3 * self.cell.n * F) / (
            n_f * self.n_cells * self.cell.M * self.dt
        )

        d_eol = ((self.max_current * V_cell_bol) / i_eol) - V_cell_bol
        return d_eol

    def estimate_time_until_replacement(self):
        """Estimate the time until replacement based on fraction of life used,
        which is the ratio of the stack degradation to the end of life degradation.

        Returns:
            float: Number of hours until stack should be replaced with respect to
            simulation duration (alternatively, with respect to the number of
            hours the stack has existed in the plant.)
        """

        frac_of_life_used = self.V_degradation / self.d_eol
        # time between replacement [hrs] based on time its existed (whether on or off)
        time_between_replacement = (1 / frac_of_life_used) * (self.time / 3600)  # [hrs]
        return time_between_replacement

    def estimate_stack_life(self):
        """Estimate the stack life based on fraction of life used, which is the ratio
        of the stack degradation to the end of life degradation.

        Returns:
            float: Stack life in hours with respect to number of hours the stack has
            been operational.
        """
        # stack life [hrs] based on time its been operational
        frac_of_life_used = self.V_degradation / self.d_eol
        stack_life = (1 / frac_of_life_used) * (self.uptime / 3600)  # [hrs]
        return stack_life

    def estimate_life_performance_from_year(self, plant_life_years: int):
        """Estimate future performance of the stack assuming the
        same operation from the simulation for the duration of the plant life.

        Note:
            This function is not tested and may only work for simulations
            with an hourly timestep and a simulation length of 8760 hours.

        Args:
            plant_life_years (int): number of years in the plant life.

        Returns:
            3-element tuple containing

            - **refurb_schedule** (list): refurbishment schedule of the stack,
                a value of 1 represents a year that the stack has to be replaced.
            - **ahp_kg** (list): annual hydrogen production of the stack per year
                of the ``plant_life_years``. Each element has units of kg-H2/year
            - **aep_kWh** (list): annual energy consumption of the stack per year
                of the ``plant_life_years``. Each element has units of kWh/year.
        """
        refurb_schedule = np.zeros(plant_life_years)
        ahp_kg = np.zeros(plant_life_years)
        aep_kWh = np.zeros(plant_life_years)

        V_deg = np.array(self.degradation_history)
        V_cell_bol = np.array(self.voltage_history) - V_deg
        sim_length = len(V_cell_bol)

        Vdeg0 = 0
        for i in range(plant_life_years):
            V_deg_pr_sim = Vdeg0 + V_deg
            if np.max(V_deg_pr_sim) > self.d_eol:
                idx_dead = np.argwhere(V_deg_pr_sim > self.d_eol)[0][0]
                V_deg_pr_sim = np.concatenate(
                    [V_deg_pr_sim[0:idx_dead], V_deg[idx_dead:sim_length]]
                )
                refurb_schedule[i] = 1
            if self.hydrogen_degradation_penalty:
                I_nom = self.electrolyzer_model(
                    (self.power_input_history / 1e3, self.temperature), *self.fit_params
                )
                V_cell = self.cell.calc_cell_voltage(I_nom, self.temperature)
                eff_mult = (V_cell + self.V_degradation) / V_cell  # 1 + efficiency drop
                I_stack = I_nom / eff_mult

            else:
                I_stack = self.electrolyzer_model(
                    (self.power_input_history / 1e3, self.temperature), *self.fit_params
                )
                V_cell = self.cell.calc_cell_voltage(I_stack, self.temperature)

            H2_mass_out = (
                self.cell.calc_mass_flow_rate(self.temperature, I_stack)
                * self.n_cells
                * self.dt
            )
            power_usage_kW = self.calc_stack_power(I_stack, V_cell + V_deg_pr_sim)
            ahp_kg[i] = np.sum(H2_mass_out)
            aep_kWh[i] = np.sum(power_usage_kW)
            Vdeg0 = V_deg_pr_sim[sim_length - 1]
        return refurb_schedule, ahp_kg, aep_kWh
