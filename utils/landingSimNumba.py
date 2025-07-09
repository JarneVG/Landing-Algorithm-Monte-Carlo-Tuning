import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from numba import njit
from collections import namedtuple
try:
    from . import thrustCurves
except ImportError:
    import thrustCurves


# Later load very quickly:
times = np.load("times_10deg.npy")
angle_of_thrust = np.load("angles_10deg.npy")
angle_of_thrust_rad = np.radians(angle_of_thrust)

def shift_array(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts the elements of an array by a specified number of positions.
    
    Parameters:
        arr (np.ndarray): The input array to be shifted.
        shift (int): The number of positions to shift the array. Positive values shift right, negative values shift left.
        
    Returns:
        np.ndarray: The shifted array.
    """
    if shift > 0:
        return np.concatenate((arr[0]*np.ones(shift), arr[:-shift]))
    elif shift < 0:
        return np.concatenate((arr[-shift:], arr[-1]*np.ones(-shift)))
    else:
        return arr.copy()
    
def extend_array(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Extends an array to a specified length by repeating its last element.
    
    Parameters:
        arr (np.ndarray): The input array to be extended.
        length (int): The desired length of the output array.
        
    Returns:
        np.ndarray: The extended array.
    """
    if len(arr) >= length:
        return arr[:length]
    else:
        return np.concatenate((arr, np.full(length - len(arr), arr[-1])))
    
    
def get_vertical_thrust_array(
    motor_type: str,
    number_of_motors: int,
    angle_of_thrust_rad: np.ndarray,
    ignition_time: float,
    time_step: float = 0.01,
    max_time: float = 20.0
) -> np.ndarray:
    """
    Computes the vertical thrust array based on the motor type, number of motors, angle of thrust, and ignition time.
    
    Parameters:
        motor_type (str): The type of motor used.
        number_of_motors (int): The number of motors.
        angle_of_thrust (np.ndarray): The angle of thrust in degrees.
        ignition_time (float): The time at which the motors ignite.
        time_step (float): The time step for the simulation.
        
    Returns:
        np.ndarray: An array representing the vertical thrust over time.
    """
    
    # First shift based on ignition time
    thrust_array = number_of_motors*thrustCurves.get_thrust_array(motor_type, np.arange(0, max_time, time_step))
    percentage_of_thrust_vertical = np.cos(angle_of_thrust_rad)  # Vertical component of thrust
    percentage_of_thrust_vertical = extend_array(percentage_of_thrust_vertical, int(max_time / time_step))

    vertical_thrust = thrust_array * percentage_of_thrust_vertical
    
    shift = int(ignition_time / time_step)
    vertical_thrust_shifted = shift_array(vertical_thrust, shift)
    
    # Placeholder for actual thrust computation logic
    # This should be replaced with actual data or a model
    return vertical_thrust_shifted

LandingResult = namedtuple("LandingResult", ["velocity", "landing_status", "remaining_burn_time", "equivalent_drop_height", "time_of_landing"])

@njit
def fast_landing_sim(
    vertical_thrust_array = np.array([]), # This is a precomputed thrust array based on the motor type, num of motors, angle of thrust and ignition time
    mass_array = np.array([]),
    frontal_area = np.pi*0.035**2,
    Cd = 0.82,
    max_equivalent_drop_height = 1.5,
    ignition_time = 1.0,
    burn_time = 6.0,
    initial_altitude = 30.0,
    initial_vertical_velocity = -0.0,
    time_step = 0.01,
    max_sim_time = 20.0,
    rho = 1.225):

    """
    Fast landing simulation using Numba for performance.
    Parameters:
        vertical_thrust_array (np.ndarray): Precomputed thrust array based on motor type, number of motors, angle of thrust, and ignition time.
        mass_array (np.ndarray): Array of mass values at each time step.
        frontal_area (float): Frontal area of the rocket.
        Cd (float): Drag coefficient.
        max_equivalent_drop_height (float): Maximum equivalent drop height the rocket can handle.
        initial_altitude (float): Initial altitude of the rocket.
        initial_vertical_velocity (float): Initial vertical velocity of the rocket.
        time_step (float): Time step for the simulation.
        max_sim_time (float): Maximum simulation time.
    Returns:
        None: The function modifies the altitude and velocity in place.
    """
    motor_burnout = False
    landing_status = 'too late'
    remaining_burn_time = 1000 # Placeholder for remaining burn time
    equivalent_drop_height = 1000.0 # Placeholder for equivalent drop height
    numOfSteps = int(max_sim_time / time_step)
    velocity = initial_vertical_velocity
    altitude = initial_altitude
    for i in range(numOfSteps):
        # Calculate forces
        drag_force = 0.5 * rho * Cd * frontal_area * (velocity ** 2) * np.sign(velocity)
        net_force = vertical_thrust_array[i] - drag_force - mass_array[i]*9.81 # drag force goes with minus because of sign(velocity)
        
        acceleration = net_force / mass_array[i]  # Thrust - gravity
        velocity += acceleration * time_step
        altitude += velocity * time_step
    
        # if you hit the ground, stop sim and store data
        if altitude < 0:
            altitude = 0
            # Check if motor has burned out
            t = i * time_step

            if (t - ignition_time) >= burn_time-2: # check if you're at least 2 seconds before the end burnout to eliminate false positives
                if vertical_thrust_array[i] < mass_array[i]*9.81:
                    motor_burnout = True
                    
            remaining_burn_time = ignition_time + burn_time - t
            equivalent_drop_height = (velocity ** 2) / (2 * 9.81)
            
            if motor_burnout and equivalent_drop_height < max_equivalent_drop_height:
                landing_status = 'success'
            elif motor_burnout and equivalent_drop_height >= max_equivalent_drop_height:
                landing_status = 'too early'
            
            elif not motor_burnout:
                landing_status = 'too late'
            
            break
            
    return LandingResult(velocity, landing_status, remaining_burn_time, equivalent_drop_height, t)

          
def do_fast_landing_simulation(
    angle_of_thrust_rad: np.ndarray = angle_of_thrust_rad,
    motor_type: str = 'Klima D3',
    number_of_motors: int = 2.0,
    beginning_mass: float = 0.57,
    ignition_delay: float = 0.8,
    dt: float = 0.01,
    max_time: float = 20.0,
    initial_altitude: float = 30.0,
    initial_velocity: float = 0.0,
    frontal_area: float = 0.035**2 * np.pi,
    drag_coefficient: float = 0.82,
    max_drop_height: float = 1.5,
    rho: float = 1.225
):
    """
    Prepares input arrays and runs the fast landing simulation.
    
    Parameters:
        motor_type: Rocket motor identifier (e.g. 'Klima D3').
        number_of_motors: Number of motors in the rocket.
        angle_of_thrust_rad: Thrust angle in radians.
        ignition_delay: Time delay before ignition (seconds).
        dt: Sample time (seconds).
        max_time: Maximum simulation time (seconds).
        initial_altitude: Starting altitude (meters).
        initial_velocity: Starting vertical velocity (m/s).
        frontal_area: Rocket frontal area (meters²).
        drag_coefficient: Drag coefficient (dimensionless).
        max_drop_height: Max equivalent drop height (meters).
        
    Returns:
        Simulation result from fast_landing_sim.
    """
    
    # Calculate the delay in samples corresponding to ignition delay time
    comeup_time_delay_samples = int(ignition_delay / dt)

    # Get vertical thrust time series for the motor(s)
    vertical_thrust_array = get_vertical_thrust_array(
        motor_type=motor_type,
        number_of_motors=number_of_motors,
        angle_of_thrust_rad=angle_of_thrust_rad,
        ignition_time=ignition_delay,
        time_step=dt,
        max_time=max_time
    )
    time_data = np.arange(0, max_time, dt)

    # Initial mass of the rocket (without propellant)
    
    # Get propellant mass loss curve from thrust curves
    lost_mass_array = number_of_motors * thrustCurves.get_propellant_mass_array(motor_type, time_data)
    
    # Adjust initial mass by subtracting the initial lost mass (usually zero, but safe to do)
    beginning_mass -= lost_mass_array[0]
    
    # Shift lost mass array to account for ignition delay (simulate no propellant loss before ignition)
    lost_mass_array = shift_array(lost_mass_array, comeup_time_delay_samples)
    
    # Create the full mass array over time
    mass_array = np.ones(int(max_time / dt)) * beginning_mass
    mass_array += lost_mass_array
    
    # Run the landing simulation with prepared parameters and arrays
    result = fast_landing_sim(
        vertical_thrust_array=vertical_thrust_array,
        mass_array=mass_array,
        frontal_area=frontal_area,
        Cd=drag_coefficient,
        max_equivalent_drop_height=max_drop_height,
        ignition_time=ignition_delay,
        burn_time=thrustCurves.get_burn_time(motor_type),
        initial_altitude=initial_altitude,
        initial_vertical_velocity=initial_velocity,
        time_step=dt,
        max_sim_time=max_time,
        rho=rho
        
    )

    return result

  
            
            
# Example usage of all functions
if __name__ == "__main__":
    plot_shift = False
    plot_get_propellant_mass = False
    plot_extend_array = False
    plot_vertical_thrust = False
    plot_thrust_angle = False
    run_fast_landing_sim = True
    
    
    import matplotlib.pyplot as plt
    dt = 0.002  # Sample time
    max_time = 20.0  # Maximum simulation time
    number_of_samples = int(max_time / dt)
    delay = 200  # Shift delay in number of samples (simulate start of burn)
    number_of_motors = 2
    time_data = np.arange(0, max_time, dt)
    # Example data
    if plot_shift:
        thrust_array = number_of_motors*thrustCurves.get_thrust_array('Klima D3', time_data)
        shifted_thrust_array = shift_array(thrust_array, delay)
        fig, ax = plt.subplots()
        ax.plot(thrust_array, label='Original Array')
        ax.plot(shifted_thrust_array, label='Shifted Array by 200')
        ax.legend()
        plt.show()
    
    
    if plot_get_propellant_mass:
        beginning_mass = 0.57
        lost_mass_array = number_of_motors*thrustCurves.get_propellant_mass_array('Klima D3', time_data)
        beginning_mass -= lost_mass_array[0]  # Adjust beginning mass by the initial lost mass
        lost_mass_array = shift_array(lost_mass_array, delay)
        
        mass_array = np.ones(int(max_time/dt)) * beginning_mass
        mass_array += lost_mass_array
        fig, ax = plt.subplots()
        ax.plot(mass_array, label='Mass Array')
        ax.legend()
        plt.show()
    
    
    # Plot an extended array
    if plot_extend_array:
        array = np.array([1, 2, 3, 4, 5])
        extended_array = extend_array(array, 3)
        fig, ax = plt.subplots()
        ax.plot(array, label='Original Array')
        ax.plot(extended_array, label='Extended Array', linestyle='--')
        ax.legend()
        plt.show()
    
    
    
    # Plot vertical thrust vs normal thrust
    if plot_vertical_thrust:
        vertical_thrust_array = get_vertical_thrust_array(
            motor_type='Klima D3',
            number_of_motors=2,
            angle_of_thrust_rad=angle_of_thrust_rad,
            ignition_time=1.0,
            time_step=dt
        )
        normal_thrust_array = number_of_motors*thrustCurves.get_thrust_array('Klima D3', time_data)
        fig, ax = plt.subplots()
        ax.plot(vertical_thrust_array, label='Vertical Thrust')
        ax.plot(normal_thrust_array, label='Normal Thrust')
        ax.legend()
        plt.show()
    
    if plot_thrust_angle:
        # Plot the angle of thrust
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(angle_of_thrust_rad), label='Angle of Thrust (deg)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        plt.show()
        
        
    if run_fast_landing_sim:
        import time
        start_time = time.perf_counter()
        comeup_time = .708
        comeup_time_delay_samples = int(comeup_time / dt)
        # Example parameters for the fast landing simulation
        vertical_thrust_array = get_vertical_thrust_array(
            motor_type='Klima D3',
            number_of_motors=2,
            angle_of_thrust_rad=angle_of_thrust_rad,
            ignition_time=comeup_time,
            time_step=dt
        )
        
        beginning_mass = 0.57
        lost_mass_array = number_of_motors*thrustCurves.get_propellant_mass_array('Klima D3', time_data)
        beginning_mass -= lost_mass_array[0]  # Adjust beginning mass by the initial lost mass
        lost_mass_array = shift_array(lost_mass_array, comeup_time_delay_samples)
        
        mass_array = np.ones(int(max_time/dt)) * beginning_mass
        mass_array += lost_mass_array
        
        # Run the fast landing simulation
        result = fast_landing_sim(
            vertical_thrust_array=vertical_thrust_array,
            mass_array=mass_array,
            frontal_area=np.pi*0.035**2,
            Cd=0.82,
            max_equivalent_drop_height=1.5,
            ignition_time=comeup_time,
            burn_time=thrustCurves.get_burn_time('Klima D3'),
            initial_altitude=30.0,
            initial_vertical_velocity=-0.0,
            time_step=dt,
            max_sim_time=max_time
        )
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"Simulation ran in {elapsed_time:.6f} seconds")

        # Print the results
        print("Fast Landing Simulation Results:")
        print(f"Landing Velocity: {result.velocity:.2f} m/s")
        print(f"Landing Status: {result.landing_status}")
        print(f"Remaining Burn Time: {result.remaining_burn_time:.2f} s")
        print(f"Equivalent Drop Height: {result.equivalent_drop_height:.2f} m")
        print(f"Time of Landing: {result.time_of_landing:.4f} s")
        
        
                # Usage example:
        max_time = 20.0      # simulation duration (s)
        start_time = time.perf_counter()

        result = do_fast_landing_simulation(
            motor_type='Klima D3',
            number_of_motors=2,
            beginning_mass=0.57,  # initial mass of the rocket without propellant (kg)
            angle_of_thrust_rad=angle_of_thrust_rad,
            ignition_delay=comeup_time,
            dt=dt,
            max_time=max_time,
            initial_altitude=30.0,  # initial altitude (m)
            initial_velocity=0.0,  # initial vertical velocity (m/s)
            frontal_area=0.035**2 * np.pi,  # frontal diameter of the rocket (m^2)
            drag_coefficient=0.82,  # drag coefficient
            max_drop_height=1.5  # maximum equivalent drop height (m)
        )
        
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"Second ran in {elapsed_time:.6f} seconds")

                # Print the results
        print("Fast Landing Simulation Results Clean:")
        print(f"Landing Velocity: {result.velocity:.2f} m/s")
        print(f"Landing Status: {result.landing_status}")
        print(f"Remaining Burn Time: {result.remaining_burn_time:.2f} s")
        print(f"Equivalent Drop Height: {result.equivalent_drop_height:.2f} m")
        print(f"Time of Landing: {result.time_of_landing:.4f} s")
