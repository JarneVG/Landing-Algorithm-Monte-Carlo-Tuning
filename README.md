# About

This code is meant to tune the landing algorithm for propulsively landing a model rocket and works as an onion with various layers.

The outer 2 layers go over various altitudes and vertical velocities as starting conditions and will try to make the rocket land by varying the landing burn ignition time, which is itself a loop and runs a 1DOF altitude simulation.

It handles UI of the outer onion layer with the streamlit module, run with command: 'streamlit run MonteCarloLandingSim.py'. To see if the sub python programs are running, simply run them seperately