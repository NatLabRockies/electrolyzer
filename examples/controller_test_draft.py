# run through the different types of plant control schemes

import numpy as np

from electrolyzer.simulation.supervisor import Supervisor


# import matplotlib.pyplot as plt


# Define single turbine parameters
turbine_rating = 3.4  # MW
electrolyzer_rating = 0.5  # MW

# Define Electrolyzer input dictionary
n_stacks = round(turbine_rating / electrolyzer_rating)

supervisor_dict = {}
DT = 1
supervisor_dict["dt"] = DT

electrolyzer_dict = {}
supervisor_dict["n_stacks"] = n_stacks
electrolyzer_dict["n_cells"] = 100
electrolyzer_dict["cell_area"] = 1000
electrolyzer_dict["stack_rating_kW"] = 500
electrolyzer_dict["max_current"] = 2000
# electrolyzer_dict["stack_input_voltage"] = 250
electrolyzer_dict["temperature"] = 60
electrolyzer_dict["dt"] = DT

supervisor_dict["stack"] = electrolyzer_dict

# Test Seven Hydrogen Plant Control Strategy Options
control_type_vec = [
    "BaselineDeg",
    "PowerSharingRotation",
    "SequentialRotation",
    "EvenSplitEagerDeg",
    "EvenSplitHesitantDeg",
    "SequentialEvenWearDeg",
    "SequentialSingleWearDeg",
]
Nc = len(control_type_vec)

# Create cosine test signal
test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
base_value = (turbine_rating / 2) + 0.2
variation_value = turbine_rating - base_value
power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

# ######### Plot input signal #########
# plt.figure()
# plt.plot(power_test_signal/ 1e6)
# plt.show()

for i in range(Nc):
    # Define controller type
    control_type = control_type_vec[i]
    supervisor_dict["control_type"] = control_type
    print("Controller Type: ", control_type)

    # Initialize Electrolyzer system
    elec_sys = Supervisor.from_dict(supervisor_dict)

    # Define output variables
    kg_rate = np.zeros((elec_sys.n_stacks, len(power_test_signal)))
    degradation = np.zeros((elec_sys.n_stacks, len(power_test_signal)))
    uptime = np.zeros((elec_sys.n_stacks))
    wind_curtailment = np.zeros((len(power_test_signal)))
    tot_kg = np.zeros((len(power_test_signal)))
    kg_rate = np.zeros((n_stacks, len(power_test_signal)))
    p_in = []

    # Run electrolyzer simulation
    for i in range(len(power_test_signal)):
        # if (i % 1000) == 0:
        #     print('Progress', i)
        # print(i)
        loop_H2, loop_h2_mfr, loop_power_left, curtailed_wind = elec_sys.control(
            power_test_signal[i]
        )
        p_in.append(power_test_signal[i] / elec_sys.n_stacks / 1000)

        tot_kg[i] = np.copy(loop_H2)
        wind_curtailment[i] = np.copy(curtailed_wind) / 1000000
        for j in range(n_stacks):
            kg_rate[j, i] = loop_h2_mfr[j]
            degradation[j, i] = elec_sys.stacks[j].V_degradation

    # Print output variables
    for i in range(n_stacks):
        print("degradation", elec_sys.stacks[i].V_degradation)
        print("fatigue", elec_sys.stacks[i].fatigue_history)
        print("cycles", elec_sys.stacks[i].cycle_count)
        print("uptime", elec_sys.stacks[i].uptime)
        uptime[i] = elec_sys.stacks[i].uptime

    print("Total kgs of H2 produced: ", sum(tot_kg))
    print("Total wind curtailed: ", sum(wind_curtailment))

    # ###### Plot key variables in time ######

    # plt.figure(figsize=[8,4])
    # plt.plot(tot_kg)
    # plt.title('kg produced')
    # plt.xlabel('Time [s]')
    # plt.ylabel('H2 [kg]')
    # plt.ylim(-0.002,0.016)

    # plt.figure(figsize=[8,4])
    # plt.plot(p_in)
    # plt.title('Power provided to each electrolyzer')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Power [MW]')

    # plt.figure(figsize=[8,4])
    # plt.plot(np.transpose(kg_rate))
    # plt.title('kg production rate')
    # plt.xlabel('Time [s]')
    # plt.ylabel('H2 rate [kg/s]')
    # plt.ylim(-0.0002,0.0022)

    # plt.figure(figsize=[8,4])
    # plt.plot(np.transpose(degradation))
    # plt.title('Electrolyzer Degradation')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Degradation [V]')
    # plt.ylim(-0.001,0.025)

    # plt.figure(figsize=[8,4])
    # plt.plot(np.transpose(wind_curtailment))
    # plt.title('Power curtailed')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Power [MW]')
    # plt.ylim(-0.2,3.5)

    # plt.figure(figsize=[8,4])
    # plt.plot(range(0,n_stacks), (uptime/len(power_test_signal))*100, '*')
    # plt.title('Uptime percentage each electrolyzer')
    # plt.xlabel('Electrolyzer Number')
    # plt.ylabel('Utilization Percent')
    # plt.ylim(-5,105)
    # plt.show()
