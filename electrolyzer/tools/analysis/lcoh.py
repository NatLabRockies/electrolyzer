"""
This module defines the Hydrogen Electrolyzer control code.
"""

import numpy as np
import pandas as pd
from attrs import field, define

from electrolyzer.tools.type_dec import NDArrayFloat, FromDictMixin


@define
class LCOH(FromDictMixin):
    # Cost Parameters #
    ####################

    plant_params: dict
    feedstock: dict
    opex: dict
    stack_replacement: dict
    capex: dict
    finances: dict

    # Simulation result details
    dt: float
    sim_length_hrs: float
    plant_rating_kW: float
    n_stacks: int
    stack_rating_kW: float
    deg_state: NDArrayFloat
    electrical_feedstock_cost: float
    power_kW_avail: NDArrayFloat
    power_kW_curtailed: NDArrayFloat
    kg_produced: NDArrayFloat

    plant_life: int = field(init=False, default=25)
    pem_location: str = field(init=False, default="onshore")
    water_feedstock_cost: float = field(init=False)

    capex_summary: dict = field(init=False)
    opex_summary: dict = field(init=False)
    feedstock_summary: dict = field(init=False)
    stack_replacement_summary: dict = field(init=False)
    stack_replacement_schedule: dict = field(init=False)
    LCOH_summary: dict = field(init=False)

    os: int = field(init=False, default=0)
    IF: float = field(init=False, default=0.33)
    DR: float = field(init=False, default=0.04)
    plant_life_yrs: float = field(init=False, default=25)

    def __attrs_post_init__(self) -> None:
        """
        --- Current LCOH Flow ---

        Inputs are user-defined under "cost" in modeling_inputs.yaml file
        Annual Costs:
            OpEx:
                functions used:
                    -calc_OM_costs (main)
                    -calc_power_based_capacity_factor (called in main)
                -Variable O&M: depends on plant capacity factor
                    -user_inputs:
                        var_OM
                        fixed_OM
                        stack_rating_kW
                        n_stacks
                        fixed_OM
                    -simulation_data:
                        'power_signal'
                        'curatilment'
                -Fixed O&M: annual cost regardless of operation
                    -user_inputs:
                    -simulation_data:
            Feedstock:
                functions used:
                    -calc_feedstock_costs
                -user_inputs:
                    water_feedstock_cost
                    water_per_kgH2
                    dt
                -external_data:
                    lcoe
                -simulation_data:
                    'kg_rate' -> to calculate H20
                    'power_signal'
                    'curatilment'
                -simulation data scaled to represent a year
                -Water feedstock (requires H2 production)
                -Electricity Feedstock (requires LCOE)
            Stack Replacements:
                functions used:
                    -calc_stack_replacement_costs (main)
                    -calc_stack_replacement_schedule (called by main)
                -user_inputs:
                    plant_life
                    d_eol
                    n_stacks
                    stack_rating_kW
                    stack_replacement_percent
                -simulation_data:
                    deg_state
                    sim_length (time)
                -first determine stack replacement schedule
                -cost for 1 stack replacement is total uninstalled CapEx
                    divided by number of stacks
                -multiply cost by replacement schedule
                -OUPUTS: stack replacement cost per year (array of length plant_life)
        CapEx Costs:
            functions used:
                -calc_pem_capex
                -calc_bop_capex
            -split into BOP and PEM
                -user_inputs:
                    capex_learning_rate
                    ref_cost_bop & ref_cost_pem
                    ref_size_bop & ref_size_pem
                    stack_rating_kW
                    n_stacks
                    install_factor

        H2 Production:
            -done in LCOH calculation
            -simulation data scaled to represent a year
        """
        self.capex_summary = {}
        self.stack_replacement_summary = {}
        self.stack_replacement_schedule = {}
        self.feedstock_summary = {}
        self.opex_summary = {}
        self.LCOH_summary = {}
        self.plant_life_yrs = self.plant_params["plant_life"]
        if self.plant_params["pem_location"] == "onshore":
            self.os = 0
        else:
            self.os = 1

        self.IF = self.finances["install_factor"]
        self.DR = self.finances["discount_rate"]

    def calc_pem_capex(self):
        # PEM capex using cost scaling
        lr = self.capex["capex_learning_rate"]
        b_capex = np.log2(1 - lr)
        Sr0 = self.capex["ref_size_pem"]  # ref size [kW]
        Cr0 = self.capex["ref_cost_pem"]  # ref cost [$/kW]
        # self.plant_rating_kW
        Sr = (self.stack_rating_kW / Sr0) ** b_capex  # scale sized
        adj_IF = 1 + self.IF * self.os
        Cr_uninstalled = Cr0 * Sr  # [$/kW]
        Cr_installed = adj_IF * Cr0 * Sr  # [$/kW]
        capex_pem_dollars_installed = Cr_installed * self.stack_rating_kW
        capex_pem_dollars_uninstalled = Cr_uninstalled * self.stack_rating_kW
        self.capex_summary["PEM"] = {
            "Uninstalled [$/kW]": Cr_uninstalled,
            "Installed [$/kW]": Cr_installed,
            "Uninstalled [$]": capex_pem_dollars_uninstalled,
            "Installed [$]": capex_pem_dollars_installed,
        }
        return capex_pem_dollars_installed

    def calc_bop_capex(self):
        # BOP capex using cost scaling
        lr = self.capex["capex_learning_rate"]
        b_capex = np.log2(1 - lr)
        Sr0 = self.capex["ref_size_bop"]
        Cr0 = self.capex["ref_cost_bop"]
        Sr = (self.stack_rating_kW / Sr0) ** b_capex  # scale sized
        adj_IF = 1 + self.IF * self.os
        Cr_uninstalled = Cr0 * Sr  # [$/kW]
        Cr_installed = adj_IF * Cr0 * Sr  # [$/kW]
        capex_bop_dollars_installed = Cr_installed * self.stack_rating_kW
        capex_bop_dollars_uninstalled = Cr_uninstalled * self.stack_rating_kW
        self.capex_summary["BOP"] = {
            "Uninstalled [$/kW]": Cr_uninstalled,
            "Installed [$/kW]": Cr_installed,
            "Uninstalled [$]": capex_bop_dollars_uninstalled,
            "Installed [$]": capex_bop_dollars_installed,
        }
        return capex_bop_dollars_installed

    def calc_total_capex(self):
        # NOTE: right now the capex values are calculated
        # based on stack size, not plant size then
        # multiplied by the number of stacks (sizeup_factor)
        # Alternative approach would be to use the plant_rating
        # instead and remove the scale factor

        # pem_capex = self.calc_pem_capex(self.plant_rating_kW)
        # bop_capex = self.calc_bop_capex(self.plant_rating_kW)
        sizeup_factor = self.plant_rating_kW / self.stack_rating_kW
        pem_capex = sizeup_factor * self.calc_pem_capex()
        bop_capex = sizeup_factor * self.calc_bop_capex()
        total_capex = pem_capex + bop_capex
        return total_capex

    def calc_power_based_capacity_factor(self):
        # capacity factor based on total power used by the electrolyzer
        # over its rated power consumption, would probably be better
        # to use ratio of h2 produced to rated h2 production but
        # rated h2 production isn't an accessible parameter
        #
        power_consumed_kW = (self.dt / 3600) * np.sum(
            self.power_kW_avail - self.power_kW_curtailed
        )
        rated_power_consumed = self.plant_rating_kW * self.sim_length_hrs
        plant_power_based_cf = power_consumed_kW / rated_power_consumed
        return plant_power_based_cf

    def calc_stack_replacement_costs(self):
        # this must be called after capex has been calculated
        # not using any fancy cost scaling for stack replacement base cost
        single_stack_cost_perkW = (
            self.capex_summary["PEM"]["Uninstalled [$/kW]"]
            + self.capex_summary["BOP"]["Uninstalled [$/kW]"]
        )
        stack_rep_cost = (
            self.stack_rating_kW
            * single_stack_cost_perkW
            * self.stack_replacement["stack_replacement_percent"]
        )

        n_stacks_replace_per_yr = self.calc_stack_replacement_schedule()
        annual_stack_rep_cost = n_stacks_replace_per_yr * stack_rep_cost
        self.stack_replacement_summary = {
            "Stack Replacement Cost [$/kW]": single_stack_cost_perkW,
            "Stack Replacement Cost [$/stack]": stack_rep_cost,
            "Annual Stack Replacement Cost [$]": annual_stack_rep_cost,
        }
        return annual_stack_rep_cost

    def calc_time_between_replacement(self):
        # end of life voltage value
        d_eol = self.stack_replacement["d_eol"]
        # time until death (below) [hrs]
        t_eod = (d_eol / self.deg_state) * self.sim_length_hrs
        return t_eod

    def calc_stack_replacement_schedule(self):
        # this determines what electrolyzer stacks
        # need to be replaced each year of the plant life
        # based off their degradation
        time_until_replaced = np.zeros((self.plant_life_yrs, self.n_stacks))
        n_stacks_replaced = np.zeros((self.plant_life_yrs, self.n_stacks))

        t_eod = self.calc_time_between_replacement()

        time_until_replaced[0] = t_eod
        for y in range(1, self.plant_life_yrs):
            hrs_left = time_until_replaced[y - 1] - 8760 * y
            time_until_replaced[y] = np.where(hrs_left < 0, t_eod - hrs_left, hrs_left)
            n_stacks_replaced[y] = np.where(hrs_left < 0, 1, 0)

        annual_stack_rep = np.sum(n_stacks_replaced, axis=1)
        self.stack_replacement_schedule = {
            "Hrs until Replacement": pd.DataFrame(time_until_replaced),
            "Stacks Replaced": pd.DataFrame(n_stacks_replaced),
            "Annual Number of Stacks to Replace": annual_stack_rep,
        }

        return annual_stack_rep

    def calc_water_feedstock_costs(self):
        # TODO: make kg_H20 output from PEM model
        kg_h2_sim = np.sum(self.kg_produced)
        kg_h20_sim = self.feedstock["water_per_kgH2"] * kg_h2_sim
        annual_h20_consumed = (8760 / self.sim_length_hrs) * kg_h20_sim
        water_feedstock = annual_h20_consumed * self.feedstock["water_feedstock_cost"]

        self.feedstock_summary.update(
            {
                "Annual H20 Cost [$]": water_feedstock,
                "Sim H20 Cost [$]": kg_h20_sim * self.feedstock["water_feedstock_cost"],
            }
        )

        return water_feedstock

    def calc_elec_feedstock_costs(self):
        # TODO: make power used by electrolyzer output of electrolyzer model
        # rather than calculated here :)
        if self.plant_params["grid_connected"]:
            power_consumed_kWh_sim = (self.dt / 3600) * np.sum(
                self.power_kW_avail - self.power_kW_curtailed
            )
        else:
            power_consumed_kWh_sim = (self.dt / 3600) * np.sum(self.power_kW_avail)

        annual_power_consumed = (8760 / self.sim_length_hrs) * power_consumed_kWh_sim
        elec_feedstock = self.electrical_feedstock_cost * annual_power_consumed

        self.feedstock_summary.update(
            {
                "Annual Electricity Cost [$]": elec_feedstock,
                "Sim Electricity Cost [$]": power_consumed_kWh_sim
                * self.electrical_feedstock_cost,
            }
        )

        return elec_feedstock

    def calc_feedstock_costs(self):
        # calls each feedstock function, returns projected
        # annual h20 and electricity cost
        h20_feedstock = self.calc_water_feedstock_costs()
        elec_feedstock = self.calc_elec_feedstock_costs()
        total_annual_feedstock_cost = h20_feedstock + elec_feedstock
        self.feedstock_summary.update(
            {"Total Feedstock Cost [$]": total_annual_feedstock_cost}
        )

        return total_annual_feedstock_cost

    def calc_OM_costs(self):
        # Calculate operations and maintenence costs
        # variable O&M is operations based (required capacity factor)
        # fixed O&M is capacity based (independent of plant operation)
        CF = self.calc_power_based_capacity_factor()
        VOM = CF * self.opex["var_OM"] * 8760  # $/kW-year
        FOM = self.opex["fixed_OM"]  # $/kW-year
        OM_per_year = self.plant_rating_kW * (VOM + FOM)
        self.opex_summary = {
            "CF [-]": CF,
            "Variable O&M [$/kW-year]": VOM,
            "Fixed O&M [$/kW-year]:FOM": FOM,
            "Total [$/year]": OM_per_year,
        }
        return OM_per_year

    def calc_yearly_h2_production(self):
        kg_h2_sim = np.sum(self.kg_produced)
        annual_h2_produced = (8760 / self.sim_length_hrs) * kg_h2_sim
        return annual_h2_produced

    def run_lcoh(self):
        # RUNNER FUNCTION
        # H2 production scaled to annual amount
        # discount rate applied for each year

        annual_h2_produced = self.calc_yearly_h2_production()

        annual_h2 = np.zeros(self.plant_life_yrs)
        annual_OM = np.zeros(self.plant_life_yrs)
        annual_feedstock = np.zeros(self.plant_life_yrs)
        annual_stackrep = np.zeros(self.plant_life_yrs)
        annual_reduction = np.zeros(self.plant_life_yrs)
        # annual_cost= np.zeros(self.plant_life_yrs)

        tot_capex = self.calc_total_capex()
        OM_cost = self.calc_OM_costs()
        feedstock_cost = self.calc_feedstock_costs()
        sr_cost = self.calc_stack_replacement_costs()

        for y in range(self.plant_life_yrs):
            denom = (1 + self.DR) ** y
            annual_h2[y] = annual_h2_produced / denom
            annual_OM[y] = OM_cost / denom
            annual_feedstock[y] = feedstock_cost / denom
            annual_stackrep[y] = sr_cost[y] / denom
            annual_reduction[y] = denom
        total_annual_costs = annual_OM + annual_feedstock + annual_stackrep
        lcoh = (tot_capex + np.sum(total_annual_costs)) / np.sum(annual_h2)

        data = {
            "Annual Discount": annual_reduction,
            "H2 [kg]": annual_h2,
            "OM [$]": annual_OM,
            "feedstock [$]": annual_feedstock,
            "Stack Replacement [$]": annual_stackrep,
            "total annual cost [$]": total_annual_costs,
        }
        df_data = pd.DataFrame(data)

        tot_Data = [
            tot_capex,
            np.sum(annual_OM),
            np.sum(annual_feedstock),
            np.sum(annual_stackrep),
        ]
        tot_keys = ["CapEx", "OM", "Feedstock", "Stack Rep"]
        tot_data_per_kg = np.array(tot_Data) / np.sum(annual_h2)
        self.LCOH_summary["Yearly"] = df_data
        self.LCOH_summary["Totals [$]"] = dict(zip(tot_keys, tot_Data))
        self.LCOH_summary["Totals [$/kg-H2]"] = dict(zip(tot_keys, tot_data_per_kg))
        self.LCOH_summary["LCOH [$/kg-H2]"] = lcoh

        return lcoh
