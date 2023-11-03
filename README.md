# CACTO: Continuous Actor-Critic algorithm with Trajectory Optimization
**Files**:
- ***main*** implements CACTO with state = *[x,t]*. Inputs: test-n, system-id, seed, recover-training-flag, nb-cpus, and w-S.
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

**Systems**: 
single integrator (system-id: single_integrator), double integrator (system-id: double_integrator), car (system-id: 'car'), car_park (system-id: 'car_park'), and 3 DOF planar manipulator (system-id: manipulator)

**Inputs**:
| Argument Name           | Type   | Default | Choices                                                                                              | Help                                |
|-------------------------|--------|---------|------------------------------------------------------------------------------------------------------|-------------------------------------|
| `--test-n`              | int    | 0       |                                                                                                      | Test number                         |
| `--seed`                | int    | 0       |                                                                                                      | Random and tf.random seed           |
| `--system-id`           | str    | 'single_integrator' | single_integrator, double_integrator, car, car_park, manipulator, ur5 | System-id (single_integrator, double_integrator, car, car_park, manipulator, ur5) |
| `--recover-training-flag` | bool | False | True, False | Flag to recover training |
| `--nb-cpus` | int | 2 | | Number of TO problems solved in parallel |
| `--w-S` | float | 0 | | Sobolev training - weight of the value related error |


Example of usage: 

```python3 main.py --system-id='single_integrator' --seed=0 --nb-cpus=15 --w-S=1e-2 --test-n=0```
- The "single_integrator" system is selected;
- All the seeds are set to 0;
- 15 TO problems are solved in parallel (if enough resources are available);
- The weight of the value-error is set to 1e-2 (the value-gradient-error is set to 1). Note that w-S=0 corresponds to the standard CACTO algorithm (without Sobolev-Learning);
- The information about the test and the results are stored in the folder N_try_0.