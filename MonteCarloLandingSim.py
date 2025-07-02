## Run with command: 'streamlit run MonteCarloLandingSim.py'

from utils.landingSim import simulate_landing
from utils.landingSim import RocketSpecs, LandingConditions, SimulationConfig
from utils.iterateForLandingTime import get_best_ignition_time
import streamlit as st
import numpy as np
import time

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
                    sampling_rate=timestep,
                    max_sim_time=20
                )
                # Iterate to find ignition time for this setup
                ignition_time_matrix[i,j], iterations = get_best_ignition_time(rocket, conditions, sim_config, max_iterations=max_iterations) 
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
# ...existing code...





