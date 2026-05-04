import openmdao.api as om
from attrs import field, define, validators

from electrolyzer.core.utilities import BaseConfig


@define(kw_only=True)
class CellBaseConfig(BaseConfig):
    A_cell: float = field(validator=(validators.ge(0.0)))
    membrane_thickness: float = field(validator=(validators.ge(0.0)))
    temperature: float = field(validator=(validators.ge(50.0)))
    P_anode: float = field(validator=(validators.ge(0.0)))
    P_cathode: float = field(validator=(validators.ge(0.0)))
    R_elec: float = field(validator=(validators.ge(0.0)))


class CellBaseClass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        self.dt = self.options["plant_config"]["simulation"]["dt"]

        # design variables
        self.add_input("cell_active_area", val=self.config.A_cell, units="A/(cm**2)")
        self.add_input("membrane_thickness", val=self.config.membrane_thickness, units="cm")
        self.add_input("operating_temperature", val=self.config.temperature, units="degC")
        self.add_input("anode_pressure", self.config.P_anode, units="bar")
        self.add_input("cathode_pressure", self.config.P_cathode, units="bar")
        self.add_input("R_elec", self.config.R_elec, units="ohm*(cm**2)")

        # input profiles
        self.add_input("current_in", val=0.0, shape_by_conn=True, units="A")  # OR current density?
        self.add_input("current_density_in", val=0.0, shape_by_conn=True, units="A/(cm**2)")

        # output profiles
        self.add_output("cell_voltage", val=0.0, copy_shape="current_in", units="V")
        self.add_output(
            "hydrogen_produced", val=0.0, copy_shape="current_in", units=f"kg/({self.dt}*s)"
        )
        self.add_output(
            "oxygen_produced", val=0.0, copy_shape="current_in", units=f"kg/({self.dt}*s)"
        )
        self.add_output(
            "water_consumed", val=0.0, copy_shape="current_in", units=f"kg/({self.dt}*s)"
        )
        self.add_output("hydrogen_production_rate", val=0.0, copy_shape="current_in", units="kg/s")
        self.add_input("current_density_out", val=0.0, copy_shape="current_in", units="A/(cm**2)")

        # output design variables
        self.add_output("rated_cell_voltage", val=0.0, units="V")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
