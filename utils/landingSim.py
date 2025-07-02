from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from numba import njit
try:
    from .thrustCurves import thrust_curves
except ImportError:
    from thrustCurves import thrust_curves



@dataclass
class RocketSpecs:
    motor_model: str
    number_of_motors: int
    starting_mass: float
    frontal_area: float = np.pi*0.035**2  # m^2, default frontal area
    Cd : float = 0.75  # Drag coefficient, default value for a cylinder
    max_equivalent_drop_height: float = 1.5  # m, maximum equivalent drop height for the rocket

@dataclass
class LandingConditions:
    altitude: float
    vertical_velocity: float
    starting_angle_deg: float
    
@dataclass
class SimulationConfig:
    enable_graphs: bool = True
    time_step: float = 0.01  # seconds, how often to sample the simulation data
    max_sim_time: float = 20.0  # seconds, maximum simulation time
    
@dataclass
class LandingResult:
    landing_status: str
    remaining_burn_time: float # use this as a penalty measure
    equivalent_drop_height: float
    drop_velocity: float # use this as a penalty measure
    burnout_flag: bool = False  # True if the landing motor has burned out
    
    
    
    
# === Load Thrust Angle Data ===
# Load the CSV file
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
df = pd.read_csv(os.path.join(parent_dir, "angle_data_10deg.csv"))

# Extract columns into arrays
thrust_angle_times_10deg = df["time"].values       # in seconds
thrust_angle_degrees_10deg = df["combined"].values  # in degrees

# Create interpolation function
angle_interp_func = interp1d(
    thrust_angle_times_10deg,
    thrust_angle_degrees_10deg,
    kind='linear',
    fill_value="extrapolate"  # allows querying outside time bounds
)



# === Thrust Curve Definitions ===
KlimaC2 = np.array([
    [0.0, 0.0],
    [0.04, 0.229],
    [0.12, 0.658],
    [0.211, 1.144],
    [0.291, 1.831],
    [0.385, 2.86],
    [0.447, 3.833],
    [0.505, 5.001],
    [0.567, 3.89],
    [0.615, 3.146],
    [0.665, 2.66],
    [0.735, 2.203],
    [0.815, 2.088],
    [0.93, 1.98],
    [4.589, 1.96],
    [4.729, 1.888],
    [4.815, 1.602],
    [4.873, 1.259],
    [4.969, 0.658],
    [5.083, 0.0]
])

KlimaC2_mass = np.array([
    [0.0, 11.3],
    [0.04, 11.2948],
    [0.12, 11.2544],
    [0.211, 11.1611],
    [0.291, 11.0257],
    [0.385, 10.7748],
    [0.447, 10.5387],
    [0.505, 10.2472],
    [0.567, 9.93361],
    [0.615, 9.74146],
    [0.665, 9.5763],
    [0.735, 9.38263],
    [0.815, 9.18732],
    [0.93, 8.92116],
    [4.589, 0.719051],
    [4.729, 0.412551],
    [4.815, 0.24179],
    [4.873, 0.147381],
    [4.969, 0.0426774],
    [5.083, 0.0]
])

KlimaC6 = np.array([    
    [0, 0],
    [0.046, 0.953],
    [0.168, 5.259],
    [0.235, 10.023],
    [0.291, 15.00],
    [0.418, 9.87],
    [0.505, 7.546],
    [0.582, 6.631],
    [0.679, 6.136],
    [0.786, 5.716],
    [1.26, 5.678],
    [1.357, 5.488],
    [1.423, 4.992],
    [1.469, 4.116],
    [1.618, 1.22],
    [1.701, 0.0],
])

KlimaC6_mass = np.array([
    [0.000, 9.60000],
    [0.046, 9.57895],
    [0.168, 9.21498],
    [0.235, 8.72326],
    [0.291, 8.05029],
    [0.418, 6.53342],
    [0.505, 5.80575],
    [0.582, 5.28150],
    [0.679, 4.68676],
    [0.786, 4.07772],
    [1.260, 1.48401],
    [1.357, 0.96385],
    [1.423, 0.63167],
    [1.469, 0.43046],
    [1.618, 0.04863],
    [1.701, 0.00000],
])


KlimaD3 = np.array([  
    [0, 0],
    [0.073, 0.229],
    [0.178, 0.686],
    [0.251, 1.287],
    [0.313, 2.203],
    [0.375, 3.633],
    [0.425, 5.006],
    [0.473, 6.465],
    [0.556, 8.181],
    [0.603, 9.01],
    [0.655, 6.922],
    [0.698, 5.463],
    [0.782, 4.291],
    [0.873, 3.576],
    [1.024, 3.146],
    [1.176, 2.946],
    [5.282, 2.918],
    [5.491, 2.832],
    [5.59, 2.517],
    [5.782, 1.859],
    [5.924, 1.287],
    [6.061, 0.715],
    [6.17, 0.286],
    [6.26, 0.0],
])

KlimaD3_mass = np.array([
    [0.000, 17.000],
    [0.073, 16.9921],
    [0.178, 16.947],
    [0.251, 16.8793],
    [0.313, 16.7777],
    [0.375, 16.6077],
    [0.425, 16.4047],
    [0.473, 16.146],
    [0.556, 15.5749],
    [0.603, 15.1953],
    [0.655, 14.8061],
    [0.698, 14.5559],
    [0.782, 14.1709],
    [0.873, 13.8346],
    [1.024, 13.3577],
    [1.176, 12.9226],
    [5.282, 1.61027],
    [5.491, 1.04565],
    [5.590, 0.796852],
    [5.782, 0.402106],
    [5.924, 0.192218],
    [6.061, 0.063356],
    [6.170, 0.0120934],
    [6.260, 0.000],
])


KlimaD9 = np.array([
    [0.000, 0.000],
    [0.040, 2.111],
    [0.116, 9.685],
    [0.213, 25.000],
    [0.286, 15.738],
    [0.329, 12.472],
    [0.369, 10.670],
    [0.420, 9.713],
    [0.495, 9.178],
    [0.597, 8.896],
    [1.711, 8.925],
    [1.826, 8.699],
    [1.917, 8.052],
    [1.975, 6.954],
    [2.206, 1.070],
    [2.242, 0.000],
])


KlimaD9_mass = np.array([
    [0.000, 16.1000],
    [0.040, 16.0659],
    [0.116, 15.7044],
    [0.213, 14.3477],
    [0.286, 13.1484],
    [0.329, 12.6592],
    [0.369, 12.2859],
    [0.420, 11.8667],
    [0.495, 11.2954],
    [0.597, 10.5519],
    [1.711, 2.54603],
    [1.826, 1.7287],
    [1.917, 1.11399],
    [1.975, 0.763006],
    [2.206, 0.0155338],
    [2.242, 0.0000],
])


EstesF15 = np.array([
    [0.0, 0.0],
    [0.148, 7.638],
    [0.228, 12.253],
    [0.294, 16.391],
    [0.353, 20.21],
    [0.382, 22.756],
    [0.419, 25.26],
    [0.477, 23.074],
    [0.52, 20.845],
    [0.593, 19.093],
    [0.688, 17.5],
    [0.855, 16.225],
    [1.037, 15.427],
    [1.205, 14.948],
    [1.423, 14.627],
    [1.452, 15.741],
    [1.503, 14.785],
    [1.736, 14.623],
    [1.955, 14.303],
    [2.21, 14.141],
    [2.494, 13.819],
    [2.763, 13.338],
    [3.12, 13.334],
    [3.382, 13.013],
    [3.404, 9.352],
    [3.418, 4.895],
    [3.45, 0.0]
])

EstesF15_mass = np.array([
    [0.0, 60.0],
    [0.148, 59.3164],
    [0.228, 58.3541],
    [0.294, 57.2108],
    [0.353, 55.905],
    [0.382, 55.1514],
    [0.419, 54.0771],
    [0.477, 52.3818],
    [0.52, 51.2397],
    [0.593, 49.4767],
    [0.688, 47.3744],
    [0.855, 43.9685],
    [1.037, 40.4849],
    [1.205, 37.3989],
    [1.423, 33.5],
    [1.452, 32.9674],
    [1.503, 32.026],
    [1.736, 27.8823],
    [1.955, 24.0514],
    [2.21, 19.6652],
    [2.494, 14.8632],
    [2.763, 10.4455],
    [3.12, 4.68731],
    [3.382, 0.51289],
    [3.404, 0.215344],
    [3.418, 0.0947253],
    [3.45, 0.0]
])


KlimaC2_mass_normalized = np.copy(KlimaC2_mass)
KlimaC2_mass_normalized[:, 1] = KlimaC2_mass[:, 1] / KlimaC2_mass[0, 1]

KlimaC6_mass_normalized = np.copy(KlimaC6_mass)
KlimaC6_mass_normalized[:, 1] = KlimaC6_mass[:, 1] / KlimaC6_mass[0, 1]

KlimaD3_mass_normalized = np.copy(KlimaD3_mass)
KlimaD3_mass_normalized[:, 1] = KlimaD3_mass[:, 1] / KlimaD3_mass[0, 1]

KlimaD9_mass_normalized = np.copy(KlimaD9_mass)
KlimaD9_mass_normalized[:, 1] = KlimaD9_mass[:, 1] / KlimaD9_mass[0, 1]

EstesF15_mass_normalized = np.copy(EstesF15_mass)
EstesF15_mass_normalized[:, 1] = EstesF15_mass[:, 1] / EstesF15_mass[0, 1]

# Interpolation functions for normalized mass
mass_profiles = {
    "Klima C2": interp1d(KlimaC2_mass_normalized[:, 0], KlimaC2_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Klima C6": interp1d(KlimaC6_mass_normalized[:, 0], KlimaC6_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Klima D3": interp1d(KlimaD3_mass_normalized[:, 0], KlimaD3_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Klima D9": interp1d(KlimaD9_mass_normalized[:, 0], KlimaD9_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Estes F15": interp1d(EstesF15_mass_normalized[:, 0], EstesF15_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
}

# Change motor weights etc here, interpolation still happens according to the mass loss curve
motor_specs = {
    "Klima C2": {"propellant_mass": 0.0113, "burn_time": KlimaC2[-1, 0], "total_motor_mass": 0.0224},
    "Klima C6": {"propellant_mass": 0.0096, "burn_time": KlimaC6[-1, 0], "total_motor_mass": 0.0205},
    "Klima D3": {"propellant_mass": 0.017, "burn_time": KlimaD3[-1, 0], "total_motor_mass": 0.0279},
    "Klima D9": {"propellant_mass": 0.0161, "burn_time": KlimaD9[-1, 0], "total_motor_mass": 0.0271},
    "Estes F15": {"propellant_mass": 0.060, "burn_time": EstesF15[-1, 0], "total_motor_mass": 0.102},
    "None": {"propellant_mass": 0.0, "burn_time": 0, "total_motor_mass": 0.0},
}



def simulate_landing(rocket: RocketSpecs, conditions: LandingConditions, sim_config: SimulationConfig, engine_comeup_time) -> LandingResult:
    velocity = conditions.vertical_velocity
    altitude = conditions.altitude
    mass = rocket.starting_mass
    has_fired_landing_motor = False
    landing_motor_burnout = False
    equivalent_drop_height = 0.0
    t = 0.0
    t_max = sim_config.max_sim_time  # Maximum simulation time
    dt = sim_config.time_step  # Time step for the simulation
    n_steps = int(t_max / dt)
    landing_motor_start_time = engine_comeup_time

    landing_propellant_mass = motor_specs[rocket.motor_model]["propellant_mass"] * rocket.number_of_motors
    landing_burn_time = motor_specs[rocket.motor_model]["burn_time"]
    landing_mass_func = mass_profiles[rocket.motor_model]
    remaining_landing_fuel = landing_propellant_mass
    remaining_burn_time = landing_burn_time
    
    landing_status = 'too late'
    
    # Map landing motor names to thrust data arrays
    landing_thrust_data_dict = {
        "Klima C2": KlimaC2,
        "Klima C6": KlimaC6,
        "Klima D3": KlimaD3,
        "Klima D9": KlimaD9,
        "Estes F15": EstesF15,
        "None": np.array([[0, 0]])
    }
    landing_thrust_data = landing_thrust_data_dict[rocket.motor_model]
    landing_thrust_func = interp1d(
        landing_thrust_data[:, 0], landing_thrust_data[:, 1], bounds_error=False, fill_value=0.0    
    )

    # Initialize variables to store previous mass fractions
    prev_ascent_mass_frac = 1.0
    prev_landing_mass_frac = 1.0
    
    angle_deg = conditions.starting_angle_deg
    
    if sim_config.enable_graphs:
        time_array = np.zeros(n_steps)
        altitude_array = np.zeros(n_steps)
        velocity_array = np.zeros(n_steps)
        thrust_array = np.zeros(n_steps)
        mass_array = np.zeros(n_steps)
        angle_array = np.zeros(n_steps)
        
    # === SIMULATION LOOP ===
    for i in range(n_steps):
        t = i * dt
        
        # Landing burn logic using mass profile
        if has_fired_landing_motor and (t - landing_motor_start_time) <= landing_burn_time:
            curr_landing_mass_frac = landing_mass_func(t - landing_motor_start_time)
            burned_landing_propellant = (prev_landing_mass_frac - curr_landing_mass_frac) * landing_propellant_mass
            mass -= burned_landing_propellant
            remaining_landing_fuel -= burned_landing_propellant
            prev_landing_mass_frac = curr_landing_mass_frac
            
        
        landing_thrust = 0.0
        if has_fired_landing_motor:
            # Time since landing motor started
            t_landing = t - landing_motor_start_time
            raw_thrust = rocket.number_of_motors*landing_thrust_func(t_landing)
            angle_deg = angle_interp_func(t_landing) * conditions.starting_angle_deg / 10.0  # Scale angle based on the interpolation data
            angle_rad = np.radians(angle_deg)

            # Project thrust onto vertical axis
            vertical_thrust = raw_thrust * np.cos(angle_rad)

            # Apply vertical thrust to your physics
            landing_thrust = vertical_thrust

        
        # Check if motor has burned out
        if (t - landing_motor_start_time) >= landing_burn_time-2:
            if landing_thrust < mass*9.81:
                landing_motor_burnout = True
        
        if t >= landing_motor_start_time and not has_fired_landing_motor:
            has_fired_landing_motor = True
            prev_landing_mass_frac = landing_mass_func(0.0)
                
        drag_force = 0.5 * rocket.Cd * rocket.frontal_area * (velocity**2) * np.sign(velocity)
        net_force = landing_thrust - drag_force - mass * 9.81  # Thrust - drag - gravity
        
        acceleration = net_force / mass  # Thrust - gravity
        velocity += acceleration * dt
        altitude += velocity * dt
        
        # if you hit the ground, stop sim and store data
        if altitude < 0:
            altitude = 0
            remaining_burn_time = landing_motor_start_time + landing_burn_time - t
            equivalent_drop_height = (velocity ** 2) / (2 * 9.81)

            if landing_motor_burnout and equivalent_drop_height < rocket.max_equivalent_drop_height:
                landing_status = 'success'
            elif landing_motor_burnout and equivalent_drop_height >= rocket.max_equivalent_drop_height:
                landing_status = 'too early'
            
            elif not landing_motor_burnout:
                landing_status = 'too late'
            
            break



        if sim_config.enable_graphs:
            time_array[i] = t
            altitude_array[i] = altitude
            velocity_array[i] = velocity
            thrust_array[i] = landing_thrust
            mass_array[i] = mass
            angle_array[i] = angle_deg
    
    
    # return all variables required to know if you had a good landing
    result = LandingResult(
        landing_status=landing_status,
        remaining_burn_time=remaining_burn_time,
        equivalent_drop_height=equivalent_drop_height,
        drop_velocity=velocity,
        burnout_flag=landing_motor_burnout
    )
    

    if sim_config.enable_graphs:
        print(f"Landing {result.landing_status}")
        print(f"Remaining burn time: {result.remaining_burn_time:.2f} seconds")
        print(f"Equivalent drop height: {result.equivalent_drop_height:.2f} m") 
        # Cut the arrays to the actual length of the simulation
        time_array = time_array[:i]
        altitude_array = altitude_array[:i]
        velocity_array = velocity_array[:i]
        thrust_array = thrust_array[:i]
        mass_array = mass_array[:i]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.title('Rocket Landing Simulation')
        plt.subplot(2, 2, 1)
        plt.plot(time_array, altitude_array, label='Altitude (m)')
        plt.ylabel('Altitude (m)')
        plt.grid()
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(time_array, velocity_array, label='Vertical Velocity (m/s)', color='orange')
        plt.ylabel('Vertical Velocity (m/s)')
        plt.grid()
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(time_array, thrust_array, label='Thrust (N)', color='green')
        plt.ylabel('Thrust (N)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(time_array, mass_array, label='Mass (kg)', color='red')
        plt.ylabel('Mass (kg)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()
        
        plt.show()

        pass
    


    return result



    
    
    
    
if __name__ == "__main__":
    print("Running utils.py directly")
    # Your code that should only run when utils.py is executed directly
    rocket = RocketSpecs(
        motor_model="Klima D3",
        number_of_motors=2,
        starting_mass=0.57,
        frontal_area=np.pi * 0.035**2,  # m^2, default frontal area
        Cd=0.82,  # Drag coefficient, default value for a cylinder
    )

    conditions = LandingConditions(
        altitude=30.0,
        vertical_velocity=0.0,
        starting_angle_deg=10.0
    )

    config = SimulationConfig(
        enable_graphs=False,
        time_step=0.002,  # seconds
        max_sim_time=20.0  # seconds
    )

    import time

    start_time = time.perf_counter()
    simulate_landing(rocket, conditions, config, engine_comeup_time=0.708)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Simulation ran in {elapsed_time:.6f} seconds")
