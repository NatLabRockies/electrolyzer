import openmdao.api as om


class CellBaseClass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        self.dt = self.options["plant_config"]["simulation"]["dt"]

        # design variables
        self.add_input("cell_active_area", units="A/(cm**2)")
        self.add_input("membrane_thickness", units="cm")
        self.add_input("operating_temperature", units="degC")
        self.add_input("anode_pressure", units="bar")
        self.add_input("cathode_pressure", units="bar")
        self.add_input("R_elec", units="ohm*(cm**2)")

        # input profiles
        self.add_input("current", val=0.0, shape=self.n_timesteps, units="A")

        # output profiles
        self.add_output("cell_voltage")
        self.add_output("hydrogen_produced")
        self.add_output("oxygen_produced")
        self.add_output("water_consumed")
        self.add_output("hydrogen_production_rate")

        # output design variables

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
