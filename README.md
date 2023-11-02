# CACTO: Continuous Actor-Critic algorithm with Trajectory Optimization

- ***main*** implements CACTO with state = *[q,v,t]* (joint angles, velocities and time). Inputs: test-n (default: 0), system-id (default:'-'), TO-method (default: 'pyomo'), and seed (default: None)
- ***TO*** implements the TO problem of the selected *system* whose end effector has to reach a target state while avoiding an obstacle and ensuring low control effort. The TO problem is modelled in *CasADi* and solved with *ipopt*.
- ***RL*** implements the acotr-critic RL problem of the selected *system* whose end effector has to reach a target state while avoiding an obstacle and ensuring low control. It creates the state trajectory and controls to initialize TO.
- ***NeuralNetwork*** contains the functions to create the NN-models and to compute the quantities needed to update them.
- ***environment*** contains the functions of the selected *system* (reset, step, and get-end-effector-position functions).
- ***environment_TO*** contains the functions of the selected *system* implemented with *CasADi* (step, and get-end-effector-position functions).
- ***replay_buffer*** implements a reply buffer where to store and sample transitions. It implements also a prioritized version of the replay buffer using a segment tree structure implemented in ***segment_tree*** to efficiently calculate the cumulative probability needed to sample.
- ***robot_utils*** implements the dynamics of the selected *system* with Pinocchio.
- ***plot*** contains the plot functions
- ***system_conf*** configures the training for the selected *system*. 
- ***urdf*** contains *system* URDF file (double integrator and manipulator). 

*Systems*: single integrator (system-id: single_integrator), double integrator (system-id: double_integrator), car (system-id: 'car'), car_park (system-id: 'car_park'), and 3 DOF planar manipulator (system-id: manipulator)
