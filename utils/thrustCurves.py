import numpy as np
from numba import njit



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


thrust_curves = {
    'Klima C2': KlimaC2,
    'KlimaC2 mass': KlimaC2_mass,
    'Klima C6': KlimaC6,
    'KlimaC6 mass': KlimaC6_mass,
    'Klima D3': KlimaD3,
    'KlimaD3 mass': KlimaD3_mass,
    'Klima D9': KlimaD9,
    'KlimaD9 mass': KlimaD9_mass,
    'Estes F15': EstesF15,
    'EstesF15 mass': EstesF15_mass
}

# Normalization function for mass profiles
@njit
def normalize_mass_curve(mass_array):
    normalized = np.copy(mass_array)
    normalized[:, 1] = mass_array[:, 1] / mass_array[0, 1]
    return normalized

KlimaC2_mass_normalized = normalize_mass_curve(KlimaC2_mass)
KlimaC6_mass_normalized = normalize_mass_curve(KlimaC6_mass)
KlimaD3_mass_normalized = normalize_mass_curve(KlimaD3_mass)
KlimaD9_mass_normalized = normalize_mass_curve(KlimaD9_mass)
EstesF15_mass_normalized = normalize_mass_curve(EstesF15_mass)



mass_profiles_data = {
    "Klima C2": (KlimaC2_mass_normalized[:, 0], KlimaC2_mass_normalized[:, 1]),
    "Klima C6": (KlimaC6_mass_normalized[:, 0], KlimaC6_mass_normalized[:, 1]),
    "Klima D3": (KlimaD3_mass_normalized[:, 0], KlimaD3_mass_normalized[:, 1]),
    "Klima D9": (KlimaD9_mass_normalized[:, 0], KlimaD9_mass_normalized[:, 1]),
    "Estes F15": (EstesF15_mass_normalized[:, 0], EstesF15_mass_normalized[:, 1]),
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


def get_burn_time(motor_name: str) -> float:
    """
    Returns the burn time (seconds) for the given motor.
    If motor_name is unknown, returns 0.0 seconds.
    
    Parameters:
      motor_name: str -- name of the motor
    
    Returns:
      float -- burn time in seconds
    """
    if motor_name not in motor_specs:
        return 0.0  # default burn time if unknown motor
    return motor_specs[motor_name]["burn_time"]

@njit
def fast_interp(t, xp, fp):
    return np.interp(t, xp, fp)

# Utility function example: get mass fraction at time t for given motor
def get_mass_fraction(motor_name, t):
    if motor_name not in mass_profiles_data:
        return 1.0  # default if unknown motor
    xp, fp = mass_profiles_data[motor_name]
    # np.interp returns linear interpolation; for out-of-bounds we specify left=1.0 (full mass), right=0.0 (empty)
    return fast_interp(t, xp, fp)

# Utility function example: get actual motor mass at time t
def get_propellant_mass(motor_name, t):
    fraction = get_mass_fraction(motor_name, t)
    specs = motor_specs.get(motor_name, motor_specs["None"])
    return fraction * specs["propellant_mass"]

def get_thrust(motor_name: str, t: float) -> float:
    """
    Returns the thrust (N) at time t for the given motor.
    If motor_name is unknown, returns 0.0 N.
    
    Parameters:
      motor_name: str -- name of the motor
      t: float -- time in seconds
    
    Returns:
      float -- thrust in Newtons
    """
    if motor_name not in thrust_curves:
        return 0.0  # default thrust if unknown motor
    
    thrust_curve = thrust_curves[motor_name]
    # Interpolate thrust at time t
    return fast_interp(t, thrust_curve[:, 0], thrust_curve[:, 1])

def get_thrust_array(motor_name: str, t_array: np.ndarray) -> np.ndarray:
    """
    Returns an array of thrust values at each time in t_array for the given motor.
    If motor_name is unknown, returns an array of zeros (0 N).
    
    Parameters:
      motor_name: str -- name of the motor
      t_array: np.ndarray -- array of time samples (seconds)
    
    Returns:
      np.ndarray -- array of thrust values, same shape as t_array
    """
    if motor_name not in thrust_curves:
        return np.zeros_like(t_array)  # zero thrust by default
    
    xp, fp = thrust_curves[motor_name][:, 0], thrust_curves[motor_name][:, 1]
    # Vectorized interpolation for all time points in t_array
    thrust_array = fast_interp(t_array, xp, fp)
    return thrust_array




def get_mass_fraction_array(motor_name: str, t_array: np.ndarray) -> np.ndarray:
    """
    Returns an array of normalized mass fractions at each time in t_array for the given motor.
    If motor_name is unknown, returns an array of ones (full mass).
    
    Parameters:
      motor_name: str -- name of the motor
      t_array: np.ndarray -- array of time samples (seconds)
    
    Returns:
      np.ndarray -- array of normalized mass fractions, same shape as t_array
    """
    if motor_name not in mass_profiles_data:
        return np.ones_like(t_array)  # full mass by default
    
    xp, fp = mass_profiles_data[motor_name]
    # Vectorized interpolation for all time points in t_array
    mass_frac_array = fast_interp(t_array, xp, fp)
    return mass_frac_array


def get_propellant_mass_array(motor_name: str, t_array: np.ndarray) -> np.ndarray:
    """
    Returns an array of actual motor masses (kg) at each time in t_array for the given motor.
    Uses the normalized mass fraction and multiplies by total motor mass.
    
    Parameters:
      motor_name: str -- name of the motor
      t_array: np.ndarray -- array of time samples (seconds)
    
    Returns:
      np.ndarray -- array of motor masses, same shape as t_array
    """
    mass_frac_array = get_mass_fraction_array(motor_name, t_array)
    specs = motor_specs.get(motor_name, motor_specs["None"])
    total_mass = specs["propellant_mass"]
    return mass_frac_array * total_mass




if __name__ == "__main__":
    time = 4.5  # seconds
    mass_frac = get_mass_fraction("Klima C2", time)
    print(f"Mass fraction at {time}s: {mass_frac:.3f}")
    
    propellant_mass = get_propellant_mass("Klima C2", time)
    print(f"Motor mass at {time}s: {propellant_mass:.3f} kg")
    
    thrust = get_thrust("Klima C2", time)
    print(f"Thrust at {time}s: {thrust:.3f} N")
    
    
    # Show plots of thrust and mass curves
    import matplotlib.pyplot as plt
    t_array = np.linspace(0, 6, 100)  # time from 0 to 6 seconds
    thrust_array = get_thrust_array("Klima C2", t_array)
    mass_frac_array = get_mass_fraction_array("Klima C2", t_array)
    motor_mass_array = get_propellant_mass_array("Klima C2", t_array)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(t_array, thrust_array, label='Thrust (N)')
    plt.title('Thrust Curve for Klima C2')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.grid()
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(t_array, mass_frac_array, label='Mass Fraction', color='orange')
    plt.title('Mass Fraction Curve for Klima C2')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Fraction')
    plt.grid()
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(t_array, motor_mass_array, label='Motor Mass (kg)', color='green')
    plt.title('Motor Mass Curve for Klima C2')
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Mass (kg)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
