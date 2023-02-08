# CACTO: Continuous Actor-Critic algorithm with Trajectory Optimization

- ***main*** implements CACTO with state = *[q,v,t]* (joint angles, velocities and time). Inputs: system (default:'-'), TO-method (default: 'pyomo'), and seed (default: None)
- ***CACTO*** initializes the variables used both in CACTO.
- ***TO*** implements the TO problem of the selected *system* whose end effector has to reach a target state while avoiding an obstacle and ensuring low control effort. The TO problem is modelled in *Pyomo* and solved with *ipopt*.
- ***RL*** implements the acotr-critic RL problem of the selected *system* whose end effector has to reach a target state while avoiding an obstacle and ensuring low control effort.
- ***environment*** contains the training functions of the selected *system* (reset, step (both array and tensor version), and get-end-effector-position functions).
- ***replay_buffer*** implements a reply buffer where to store and sample transitions. It implements also a prioritized version of the replay buffer using a segment tree structure implemented in ***segment_tree***  to efficiently calculate the cumulative probability needed to sample.
- ***robot_utils*** implements the dynamics of the selected *system* with Pinocchio.
- ***plot*** contains the plot functions
- ***system_conf*** configures the training for the selected *system*. 
- ***inits*** contains the functions for the selected *system* to warm-start TO (ICS, CACTO's rollout, 0s). 
- ***urdf*** contains *system* URDF file. 

*Systems*: double integrator and 3 DOF manipulator