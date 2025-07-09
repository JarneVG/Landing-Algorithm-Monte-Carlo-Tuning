## Run with command: 'streamlit run MonteCarloLandingSim.py'

import utils.thrustCurves
from utils.landingSim import simulate_landing
from utils.landingSim import RocketSpecs, LandingConditions, SimulationConfig
from utils.iterateForLandingTime import get_best_ignition_time
import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def air_density_with_humidity(temperature_C, relative_humidity, pressure_Pa=101325):
    """
    Calculate moist air density from temperature, pressure, and relative humidity.

    Parameters:
        temperature_C (float): Temperature in °C
        relative_humidity (float): Relative humidity in % (0-100)
        pressure_Pa (float): Ambient pressure in Pascals (default = sea level)

    Returns:
        float: Moist air density in kg/m³
    """
    # Constants
    R_d = 287.05     # J/(kg·K), for dry air
    R_v = 461.495    # J/(kg·K), for water vapor

    # Convert temperature to Kelvin
    T = temperature_C + 273.15

    # Saturation vapor pressure over water (in Pa)
    p_sat = 610.78 * 10 ** (7.5 * temperature_C / (237.3 + temperature_C))

    # Actual vapor pressure (Pa)
    p_v = (relative_humidity / 100.0) * p_sat

    # Partial pressure of dry air (Pa)
    p_d = pressure_Pa - p_v

    # Air density (kg/m³)
    density = (p_d / (R_d * T)) + (p_v / (R_v * T))

    return density


st.title("Monte Carlo Landing Simulation")

# ==== setup rocket parameters =====
with st.form("Rocket Specifications"):
    st.header("Rocket Specifications")
    rocket = RocketSpecs(
        motor_model=st.selectbox(
            "Select motor model:",
            ["Klima D3", "Klima D9", "Klima C2", "Klima C6", "Estes F15" ]
        ),
        number_of_motors=st.slider(
            "Select number of motors:",
            min_value=1,
            max_value=4,
            value=2,  # default selected number
            step=1
        ),
        starting_mass=st.slider(
            "Enter mass at apogee(kg):",
            min_value=0.0,
            max_value=1.0,
            value=0.57,  # default selected mass
            step=0.001,
            format="%.3f"  # format to 3 decimal places
        ),
        Cd=st.slider(
            "Enter drag coefficient:",
            min_value=0.0,
            max_value=1.5,
            value=0.82,  # default selected drag coefficient
            step=0.01,
            format="%.2f"  # format to 2 decimal places
        ),
        frontal_area=st.slider(
            "Enter frontal area (m^2):",
            min_value=0.00,
            max_value=0.05,
            value=np.pi*0.035**2,  # default selected frontal area
            step=0.001,
            format="%.3f"  # format to 3 decimal places
        )
    )
    temperature_C = st.slider(
        "Ambient Temperature (°C)", 
        value=20.0,
        min_value=-10.0,
        max_value=50.0,
        step=1.0,
        format="%d"
    )

    relative_humidity = st.slider(
        "Relative Humidity (%)", 
        value=50.0,
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        format="%d"
    )

    rho = air_density_with_humidity(temperature_C, relative_humidity)
    st.write(f"Air Density: {rho:.2f} kg/m³")

    st.divider()
    # ==== setup landing conditions =====
    st.header("Landing Conditions")
    # Create a two-sided slider
    altitude_range = st.slider(
        "Select altitude range (m):",
        min_value=0,
        max_value=100,
        value=(15, 40),  # default selected range
        step=1
    )
    altitude_resolution = st.slider(
        "Select altitude resolution (m):",
        min_value=.1,
        max_value=5.0,
        value=1.0,  # default selected resolution
        step=.1
    )
    # Create a two-sided slider
    velocity_range = st.slider(
        "Select velocity range (m/s):",
        min_value=-25,
        max_value=0,
        value=(-15, 0),  # default selected range
        step=1
    )
    velocity_resolution = st.slider(
        "Select velocity resolution (m/s):",
        min_value=.1,
        max_value=5.0,
        value=1.0,  # default selected range
        step=.1
    )
    
    
    starting_angle_deg = st.slider(
        "Select starting angle (degrees):",
        min_value=0,
        max_value=90,
        value=10,  # default selected angle
        step=1
    )
    
    max_equivalent_drop_height = st.slider(
        "Select maximum equivalent drop height (m):",
        min_value=0.0,
        max_value=3.0,
        value=1.0,  # default selected height
        step=0.1,
        format="%.1f"  # format to 1 decimal place
    )
    max_iterations = st.slider(
        "Select maximum iterations for solver:",
        min_value=1,
        max_value=1000,
        value=50,  # default selected iterations
        step=1
    )
    timestep = st.number_input(
        "Select simulation timestep (s):",
        min_value=0.0,
        max_value=0.1,
        value=0.001,  # default selected iterations
        format="%.4f"
    )
    
    run = st.form_submit_button("Run Simulation")
    


if run:
    with st.spinner("Initializing parameters..."):
        
        # Compute the altitude and velocity points
        altitude_points = np.arange(altitude_range[0], altitude_range[1] + altitude_resolution, altitude_resolution)
        velocity_points = np.arange(velocity_range[0], velocity_range[1] + velocity_resolution, velocity_resolution)  # 1 m/s resolution

        # Initialize ignition time matrix (rows = altitudes, cols = velocities)
        ignition_time_matrix = np.zeros((len(altitude_points), len(velocity_points)))
    with st.spinner("Running simulation..."):
        altitude_progress = st.empty()
        altitude_progress_bar = st.progress(0)
        velocity_progress = st.empty()
        velocity_progress_bar = st.progress(0)
        number_of_iterations = st.empty()
        ignition_comeup_time = st.empty()

        for i in range(len(altitude_points)):
            altitude_percentage = (i / len(altitude_points)) * 100
            altitude_progress.text(f"Finding matches for an altitude of {altitude_points[i]} m [{altitude_percentage:.0f}%]")
            altitude_progress_bar.progress(altitude_percentage/100)
            for j in range(len(velocity_points)):
                velocity_percentage = (j / len(velocity_points)) * 100  
                velocity_progress.text(f"Finding matches for a velocity of {velocity_points[j]} m/s [{velocity_percentage:.0f}%]")
                velocity_progress_bar.progress(velocity_percentage/100)
                
                conditions = LandingConditions(
                    altitude=altitude_points[i],
                    vertical_velocity=velocity_points[j],
                    starting_angle_deg=starting_angle_deg
                )
                sim_config = SimulationConfig(
                    enable_graphs=False,
                    time_step=timestep,
                    max_sim_time=20
                )
                # Iterate to find ignition time for this setup
                ignition_time_matrix[i,j], iterations = get_best_ignition_time(rocket, conditions, sim_config, rho, max_iterations=max_iterations) 
                number_of_iterations.text(f"Number of iterations before convergence: {iterations}")
                ignition_comeup_time.text(f"Ignition comeup time: {ignition_time_matrix[i,j]:.2f}s")
                    
    st.success(f"Result:")


    # 3D Plot of ignition_time_matrix
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[go.Surface(
            z=ignition_time_matrix,
            x=altitude_points,
            y=velocity_points,
            colorscale='Viridis'
        )]
    )
    fig.update_layout(
        title="Ignition Time Matrix",
        scene=dict(
            xaxis_title='Altitude (m)',
            yaxis_title='Vertical Velocity (m/s)',
            zaxis_title='Ignition Time (s)'
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
    # Create meshgrid
    AltitudeMesh, VelocityMesh = np.meshgrid(altitude_points, velocity_points)

    # Flatten arrays
    altitude_flat = AltitudeMesh.flatten()
    velocity_flat = VelocityMesh.flatten()
    ignition_flat = ignition_time_matrix.flatten()

    # Filter out invalid values (e.g. -0.01 meaning "invalid")
    mask = ignition_flat != -0.01
    alt_fit = altitude_flat[mask]
    vel_fit = velocity_flat[mask]
    ign_fit = ignition_flat[mask]

    # Prepare data for 2D polynomial fitting (poly22 = second order in both x and y)
    X = np.vstack((alt_fit, vel_fit)).T
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)

    # Fit polynomial
    model = LinearRegression()
    model.fit(X_poly, ign_fit)
    coeffs = model.coef_
    intercept = model.intercept_

    # Map to Arduino-style variable names
    p00 = intercept
    p10, p01 = coeffs[1], coeffs[2]
    p20, p11, p02 = coeffs[3], coeffs[4], coeffs[5]

    def generate_arduino_constants(p00, p10, p01, p20, p11, p02):
        # Generate the Arduino-style constant declarations as a multi-line string
        return (
            f"const float p00 = {p00:.6f};\n"
            f"const float p10 = {p10:.6f};\n"
            f"const float p01 = {p01:.6f};\n"
            f"const float p20 = {p20:.6f};\n"
            f"const float p11 = {p11:.6f};\n"
            f"const float p02 = {p02:.6f};"
        )
    
    arduino_constants = generate_arduino_constants(p00, p10, p01, p20, p11, p02)

    # Display it in a copyable text area
    st.text_area("Arduino Constants", value=arduino_constants, height=150)
    
    # Evaluate the fitted surface on the full meshgrid
    A_full = AltitudeMesh.flatten()
    V_full = VelocityMesh.flatten()
    X_eval = np.vstack((A_full, V_full)).T
    X_eval_poly = poly.transform(X_eval)
    fitted_values = model.predict(X_eval_poly).reshape(AltitudeMesh.shape)

    # Plot using Plotly
    fig = go.Figure()

    # Original surface
    fig.add_trace(go.Surface(
        z=ignition_time_matrix,
        x=altitude_points,
        y=velocity_points,
        colorscale='Hot',
        opacity=0.7,
        name='Original Data',
        showscale=True
    ))

    # Fitted surface
    fig.add_trace(go.Surface(
        z=fitted_values,
        x=altitude_points,
        y=velocity_points,
        colorscale='Viridis',
        opacity=0.7,
        name='Fitted Polynomial',
        showscale=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Altitude',
            yaxis_title='Velocity',
            zaxis_title='Ignition Time',
            zaxis=dict(range=[-0.009, 0.009]),
            camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0)),
        ),
        title='Original and Fitted Polynomial Surface',
        legend=dict(x=0.8, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)