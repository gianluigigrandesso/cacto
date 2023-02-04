# CACTO: Continuous Actor-Critic algorithm with Trajectory Optimization

- ***CACTO_manipulator3DoF_pyomo*** implements CACTO with state = *[q0,q1,q2,v0,v1,v2,t]* (joint angles, velocities and time)
- ***TO_manipulator3DoF_pyomo*** implements the TO problem of a 3DoF manipulator whose end effector has to reach a target state while avoiding an obstacle and ensuring low control effort. The TO problem is modelled in *Pyomo* and solved with *ipopt*.
- ***dynamics_manipulator3DoF*** implements the dynamics of a 3DoF manipulator with Pinocchio.
- ***config_manipulator3DoF_pyomo*** configures the training for both *CACTO_3DoFManipulator* and *CACTO_3DoFManipulator*.
- ***inits*** contains the functions to warm-start TO (ICS, CACTO's rollout, 0s). 
- ***replay_buffer*** implements a reply buffer where to store and sample transitions. It implements also a prioritized version of the replay buffer using a segment tree structure implemented in ***segment_tree***  to efficiently calculate the cumulative probability needed to sample.

