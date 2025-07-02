from .landingSim import RocketSpecs, LandingConditions, SimulationConfig, LandingResult
from .landingSim import simulate_landing

def get_best_ignition_time(rocket_specs, landing_conditions, sim_config, max_iterations=50):
    """
    Function to iterate through all possible ignition times and return the best one.
    
    Args:
        rocket_specs: instance of RocketSpecs
        landing_conditions: instance of LandingConditions
        sim_config: instance of SimulationConfig
        max_iterations: int, maximum number of iterations for binary search
    
    Returns:
        best_projected_ignition_time: float
    """
    late_gain = 0.1 # multiplied by drop velocity
    early_gain = 0.1
    # Initial ignition time
    ignition_time = 0.001
    iteration = 0

    # First check: is recovery possible at all?
    results = simulate_landing(
        rocket=rocket_specs,
        conditions=landing_conditions,
        sim_config=sim_config,
        engine_comeup_time=ignition_time
    )

    if results.remaining_burn_time > 0:
        print("No recovery possible with ignition time 0")
        return -0.1, iteration  # Signal failure

    # Search loop initialization
    scale_factor = 1  # step: 1, 0.5, 0.25, ...
    previous_result = 'too early'  # assume 0 is too early
    best_projected_ignition_time = ignition_time

    while iteration < max_iterations:
        iteration += 1

        # Set ignition time in the landing_conditions instance
        results = simulate_landing(
            rocket=rocket_specs,
            conditions=landing_conditions,
            sim_config=sim_config,
            engine_comeup_time=ignition_time
        )
        
        

        print(f"Iteration {iteration}: ignition_time={ignition_time:.3f}, result={results.landing_status}")

        if results.landing_status == 'too early' and previous_result == 'too early':
            ignition_time += 1 / scale_factor
            previous_result = 'too early'

        elif results.landing_status == 'too late' and previous_result == 'too late':
            ignition_time -= 1 / scale_factor
            previous_result = 'too late'

        elif results.landing_status == 'too early' and previous_result == 'too late':
            scale_factor *= 2.5
            ignition_time += 1 / scale_factor
            previous_result = 'too early'

        elif results.landing_status == 'too late' and previous_result == 'too early':
            scale_factor *= 2.5
            ignition_time -= 1 / scale_factor
            previous_result = 'too late'

        elif results.landing_status == 'success':
            best_projected_ignition_time = ignition_time
            return best_projected_ignition_time, iteration

    print("No successful ignition time found after max iterations.")
    return -0.1, iteration  # Failed to find ignition time




# ========== TEST EXECUTION BLOCK ==========

if __name__ == "__main__":
    rocket_specs = RocketSpecs(
        motor_model="Klima D3",
        number_of_motors=2,
        starting_mass=0.57,
        max_equivalent_drop_height=0.5
    )

    landing_conditions = LandingConditions(
        altitude=16.0,
        vertical_velocity=-0.0,
        starting_angle_deg=20.0
    )

    sim_config = SimulationConfig(
        enable_graphs=False,
        sampling_rate= .002,
        max_sim_time=20
    )

    best_time = get_best_ignition_time(rocket_specs, landing_conditions, sim_config)
    print(f"\n🔧 Best ignition time computed: {best_time:.3f} s")