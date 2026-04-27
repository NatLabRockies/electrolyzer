import openmdao.api as om


class StackBaseClass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        # design variables

        # input profiles
        self.add_input("current", val=0.0, shape=self.n_timesteps, units="A")

        # output profiles

        # constraints

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
