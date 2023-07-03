import os
import math
import numpy as np
import pinocchio.casadi as cpin
from robot_utils import RobotWrapper, RobotSimulator

''' CACTO parameters '''
EP_UPDATE = 25                                                                                              # Number of episodes before updating critic and actor
NUPDATES = 385000                                                                                           # Max NNs updates
UPDATE_LOOPS = 160*np.ones(int(NUPDATES/160))                                                               # Number of updates of both critic and actor performed every EP_UPDATE episodes  
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS))                                                                # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                                  # Number of algorithm loops
NSTEPS = 100                                                                                                # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                                                 # Learning rate for the critic network (1e-3)
ACTOR_LEARNING_RATE = 1e-3                                                                                  # Learning rate for the policy network (5e-4)
REPLAY_SIZE = 2**17                                                                                         # Size of the replay buffer
BATCH_SIZE = 128                                                                                            # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                                                                      # Flag to use MC or TD(n)
if not MC:
    UPDATE_RATE = 0.001                                                                                     # Homotopy rate to update the target critic network
    nsteps_TD_N = int(NSTEPS/4)                                                                             # Number of lookahed steps if TD(n) is used


### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  6000                                                                                   # Save NNs interval
else:
    save_interval = np.inf                                                                                  # Save NNs interval

plot_flag = 1
if save_flag:
    plot_rollout_interval = 400                                                                             # plot.rollout() interval (# update)
    plot_rollout_interval_diff_loc = 6000                                                                   # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval = np.inf                                                                          # plot.rollout() interval (# update)
    plot_rollout_interval_diff_loc = np.inf                                                                 # plot.rollout() interval - diff_loc (# update)



### NNs parameters
critic_type = 'elu'

NH1 = 256                                                                                                   # 1st hidden layer size
NH2 = 256                                                                                                   # 2nd hidden layer size  

LR_SCHEDULE = 1                                                                                             # Flag to use a scheduler for the learning rates
boundaries_schedule_LR_C = [200*REPLAY_SIZE/BATCH_SIZE, 
                            300*REPLAY_SIZE/BATCH_SIZE,
                            400*REPLAY_SIZE/BATCH_SIZE,
                            500*REPLAY_SIZE/BATCH_SIZE]     
# Values of critic LR                            
values_schedule_LR_C = [CRITIC_LEARNING_RATE,
                        CRITIC_LEARNING_RATE/2,
                        CRITIC_LEARNING_RATE/4,
                        CRITIC_LEARNING_RATE/8,
                        CRITIC_LEARNING_RATE/16]  
# Numbers of critic updates after which the actor LR is changed (based on values_schedule_LR_A)
boundaries_schedule_LR_A = [200*REPLAY_SIZE/BATCH_SIZE,
                            300*REPLAY_SIZE/BATCH_SIZE,
                            400*REPLAY_SIZE/BATCH_SIZE,
                            500*REPLAY_SIZE/BATCH_SIZE]   
# Values of actor LR                            
values_schedule_LR_A = [ACTOR_LEARNING_RATE,
                        ACTOR_LEARNING_RATE/2,
                        ACTOR_LEARNING_RATE/4,
                        ACTOR_LEARNING_RATE/8,
                        ACTOR_LEARNING_RATE/16]  

DECAY_RATE = 1

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

EPISODE_ICS_INIT = 0                                                                                        # Episodes where ICS warm-starting is used instead of actor rollout

kreg_l1_A = 1e-2                                                                                            # Weight of L1 regularization in actor's network - kernel
kreg_l2_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - kernel
breg_l1_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - bias
breg_l2_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - bias
kreg_l1_C = 1e-2                                                                                            # Weight of L1 regularization in critic's network - kernel
kreg_l2_C = 1e-2                                                                                            # Weight of L2 regularization in critic's network - kernel
breg_l1_C = 1e-2                                                                                            # Weight of L1 regularization in critic's network - bias
breg_l2_C = 1e-2                                                                                            # Weight of L2 regularization in critic's network - bias

### Buffer parameters
prioritized_replay_alpha = 0                                                                                # α determines how much prioritization is used, set to 0 to use a normal buffer. Used to define the probability of sampling transition i --> P(i) = p_i**α / sum(p_k**α) where p_i is the priority of transition i 
prioritized_replay_beta = 0.6          
prioritized_replay_beta_iters = None                                                                        # Therefore let's exploit the flexibility of annealing the amount of IS correction over time, by defining a schedule on the exponent β that from its initial value β0 reaches 1 only at the end of learning.
prioritized_replay_eps = 1e-2                                                                               # It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
fresh_factor = 0.95                                                                                         # Refresh factor



''' Cost function parameters '''
### Obstacles parameters
XC1 = -2.0                                                                                                  # X coord center ellipse 1
YC1 = 0.0                                                                                                   # Y coord center ellipse 1
A1  = 6                                                                                                     # Width ellipse 1 
B1  = 10                                                                                                    # Height ellipse 1 
XC2 = 3.0                                                                                                   # X coord center ellipse 2 
YC2 = 4.0                                                                                                   # Y coord center ellipse 2
A2  = 12                                                                                                    # Width ellipse 2 
B2  = 4                                                                                                     # Height ellipse 2 
XC3 = 3.0                                                                                                   # X coord center ellipse 2 
YC3 = -4.0                                                                                                  # Y coord center ellipse 2
A3  = 12                                                                                                    # Width ellipse 2 
B3  = 4                                                                                                     # Height ellipse 2 
obs_param = np.array([XC1, YC1, XC2, YC2, XC3, YC3, A1, B1, A2, B2, A3, B3])                                # Obstacle parameters vector

### Weigths
w_d = 100                                                                                                   # Distance from target weight
w_u = 1                                                                                                     # Control effort weight
w_peak = 5e5                                                                                                # Target threshold weight
w_ob = 5e6                                                                                                  # Obstacle weight
w_v = 0*1e4                                                                                                   # Velocity weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])                                                            # Weights vector (tmp)
cost_weights_running  = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, w_u])                                  # Running cost weights vector
cost_weights_terminal = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, 0])                                    # Terminal cost weights vector 

### SoftMax parameters 
alpha = 50                                                                                                  # Soft abs coefficient (obstacle) 
alpha2 = 50                                                                                                 # Soft abs coefficient (peak)
soft_max_param = np.array([alpha, alpha2])                                                                  # Soft parameters vector

### Cost function parameters
offset_cost_fun = 0                                                                                         # Reward/cost offset factor
scale_cost_fun = 1e-5                                                                                       # Reward/cost scale factor (1e-5)                                                                       
cost_funct_param = np.array([offset_cost_fun, scale_cost_fun])

### Target parameters
x_des = -20.0                                                                                               # Target x position
y_des = 0.0                                                                                                 # Target y position
TARGET_STATE = np.array([x_des,y_des])                                                                      # Target position



''' Path parameters '''
test_set = 'set 0'                                                                                          # Test id  
Config_path = './Results {}/Configs/Manipulator/'.format(test_set)                                          # Configuration path
Fig_path = './Results {}/Figures/Manipulator'.format(test_set)                                              # Figure path
NNs_path = './Results {}/NNs/Manipulator'.format(test_set)                                                  # NNs path
Log_path = './Results {}/Log/Manipulator/'.format(test_set)                                                 # Log path
DictWS_path = './Results {}/DictWS/Manipulator/'.format(test_set)                                           # DictWS path
path_list = [Fig_path, NNs_path, Log_path, DictWS_path]                                                     # Path list

# Recover-training parameters
test_set_rec = None
NNs_path_rec = './Results set {}/NNs/DoubleIntegrator'.format(test_set_rec)                                 # NNs path recover training
N_try_rec = None
update_step_counter_rec = None

''' System parameters ''' 
env_RL = 0                                                                                                  # Flag RL environment: set True if RL_env and TO_env are different                                                                                 
### Robot upload data
URDF_FILENAME = "planar_manipulator_3dof.urdf" 
modelPath = os.getcwd()+"/urdf/" + URDF_FILENAME  
robot = RobotWrapper.BuildFromURDF(modelPath, [modelPath])
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()
end_effector_frame_id = 'EE'

### Dynamics parameters
dt = 0.05                                                                                                   # Timestep   

simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(robot.na)                                                                       # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'

q_init, v_init = np.array([math.pi, math.pi, math.pi]), np.zeros(robot.nv)
simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max)

### System configuration parameters
M = 0.5                                                                                                     # Link mass
l = 10                                                                                                      # Link length
Iz = (M*l**2)/3                                                                                             # Inertia around z axis passing though an end of a link of mass M and length l
x_base = -7.0                                                                                               # x coord base
y_base = 0.0                                                                                                # y coord base

### State parameters 
nb_state = robot.nq + robot.nv + 1                                                                          # State size (robot state size +1)
x_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0])                                 # State lower bound vector
x_init_min = np.array([-math.pi, -math.pi, -math.pi, -math.pi/4, -math.pi/4, -math.pi/4, 0])                # State lower bound initial configuration vector
x_max = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf])                            # State upper bound vector
x_init_max = np.array([ math.pi,  math.pi,  math.pi,  math.pi/4,  math.pi/4,  math.pi/4, (NSTEPS-1)*dt])    # State upper bound initial configuration vector
state_norm_arr = np.array([15,15,15,10,10,10,int(NSTEPS*dt)])                                               # Array used to normalize states

# initial configurations for plot.rollout()
init_states_sim = [np.array([math.pi/4,    -math.pi/8, -math.pi/8, 0.0, 0.0, 0.0, 0.0]),                             
                   np.array([-math.pi/4,   math.pi/8,  math.pi/8,  0.0, 0.0, 0.0, 0.0]),
                   np.array([math.pi/2,    0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-math.pi/2,   0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([3*math.pi/4,  0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-3*math.pi/4, 0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([math.pi/4,    0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-math.pi/4,   0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([math.pi,      0.0,        0.0,        0.0, 0.0, 0.0, 0.0])]

### Action parameters
nb_action = robot.na                                                                                        # Action size
tau_lower_bound = -200                                                                                      # Action lower bound
tau_upper_bound = 200                                                                                       # Action upper bound
u_min = tau_lower_bound*np.ones(nb_action)                                                                  # Action lower bound vector
u_max = tau_upper_bound*np.ones(nb_action)                                                                  # Action upper bound vector



### Plot parameters
fig_ax_lim = np.array([[-41, 31], [-35, 35]])                                                               # Figure axis limit [x_min, x_max, y_min, y_max]



profile = 0                                                                                                 # Profile flag