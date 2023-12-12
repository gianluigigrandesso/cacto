import os
import math
import numpy as np
import pinocchio.casadi as cpin
from robot_utils import RobotWrapper, RobotSimulator

system_id = 'ur5'

''' CACTO parameters '''
EP_UPDATE = 200                                                                                            # Number of episodes before updating critic and actor
NUPDATES = 380000                                                                                          # Max NNs updates
UPDATE_LOOPS = np.arange(1000, 50000, 3000)                                                                # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS))                                                               # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                                 # Number of algorithm loops
NSTEPS = 100                                                                                               # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                                                # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                                 # Learning rate for the policy network
REPLAY_SIZE = 2**16                                                                                        # Size of the replay buffer
BATCH_SIZE = 64                                                                                           # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                                                                      # Flag to use MC or TD(n)
if not MC:
    UPDATE_RATE = 0.001                                                                                     # Homotopy rate to update the target critic network if TD(n) is used
    nsteps_TD_N = int(NSTEPS/4)                                                                             # Number of lookahed steps if TD(n) is used


### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  5000                                                                                   # Save NNs interval
else:
    save_interval = np.inf                                                                                  # Save NNs interval

plot_flag = 1
if plot_flag:
    plot_rollout_interval = 400                                                                             # plot.rollout() interval (# update)
    plot_rollout_interval_diff_loc = 6000                                                                   # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval = np.inf                                                                          # plot.rollout() interval (# update)
    plot_rollout_interval_diff_loc = np.inf                                                                 # plot.rollout() interval - diff_loc (# update)



### NNs parameters
critic_type = 'sine'                                                                                        # Activation function - critic (either relu, elu, sine, sine-elu)

NH1 = 256                                                                                                   # 1st hidden layer size - actor
NH2 = 256                                                                                                   # 2nd hidden layer size - actor
                                                                                                  # 2nd hidden layer size  

LR_SCHEDULE = 0                                                                                             # Flag to use a scheduler for the learning rates
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

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

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
# Obstacles parameters
XC1 = 0.0                                                                                                    # X coord center ellipse 1
YC1 = 0.25                                                                                                   # Y coord center ellipse 1
ZC1 = 0.2
ell1_center = [XC1, YC1, ZC1]
A1  = 0.5                                                                                                    # Width ellipse 1 
B1  = 0.2                                                                                                    # Height ellipse 1 
C1  = 0.34
XC2 = 0.2                                                                                                    # X coord center ellipse 2 
YC2 = 0.425                                                                                                  # Y coord center ellipse 2
ZC2 = 0.2
ell2_center = [XC2, YC2, ZC2]
A2  = 0.4                                                                                                    # Width ellipse 2 
B2  = 0.14                                                                                                   # Height ellipse 2 
C2  = 0.34
XC3 = -0.2                                                                                                   # X coord center ellipse 2 
YC3 = 0.425                                                                                                  # Y coord center ellipse 2
ZC3 = 0.2
ell3_center = [XC3, YC3, ZC3]
A3  = 0.4                                                                                                    # Width ellipse 2 
B3  = 0.14                                                                                                   # Height ellipse 2 
C3  = 0.34 
obs_param = np.array([XC1, YC1, ZC1, XC2, YC2, ZC2, XC3, YC3, ZC3, A1, B1, C1, A2, B2, C2, A3, B3, C3])     # Obstacle parameters vector

# Weigths
w_d = 100                                                                                                   # Distance from target weight
w_u = 1                                                                                                     # Control effort weight
w_peak = 5e5                                                                                                # Target threshold weight
w_ob = 5e6                                                                                                  # Obstacle weight
w_v = 0                                                                                                     # Velocity weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])                                                            # Weights vector
cost_weights_running  = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, w_u])                                  # Running cost weights vector
cost_weights_terminal = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, 0])                                    # Terminal cost weights vector 

# SoftMax parameters 
alpha = 50                                                                                                  # Soft abs coefficient (obstacle) 
alpha2 = 5                                                                                                  # Soft abs coefficient (peak)
soft_max_param = np.array([alpha, alpha2])                                                                  # Soft parameters vector

# Cost function parameters
offset_cost_fun = 0                                                                                         # Reward/cost offset factor
scale_cost_fun = 1e-5                                                                                       # Reward/cost scale factor (1e-5)                                                                       
cost_funct_param = np.array([offset_cost_fun, scale_cost_fun])

# Target parameters
x_des = 0.0                                                                                                # Target x position
y_des = 0.425                                                                                                 # Target y position
z_des = 0.2
TARGET_STATE = np.array([x_des, y_des, z_des])                                                                      # Target position



''' Path parameters '''     
test_set = 'set test'                                                                                          # Test id  
Config_path = './Results UR5/Results {}/Configs/'.format(test_set)                            # Configuration path
Fig_path = './Results UR5/Results {}/Figures'.format(test_set)                                # Figure path
NNs_path = './Results UR5/Results {}/NNs'.format(test_set)                                    # NNs path
Log_path = './Results UR5/Results {}/Log/'.format(test_set)                                   # Log path
Code_path = './Results UR5/Results {}/Code/'.format(test_set)                                 # Code path
DictWS_path = './Results UR5/Results {}/DictWS/'.format(test_set)                             # DictWS path
path_list = [Fig_path, NNs_path, Log_path, Code_path, DictWS_path]                                          # Path list

# Recover-training parameters
test_set_rec = None
NNs_path_rec = './Results UR5/Results set {}/NNs'.format(test_set_rec)                                 # NNs path recover training
N_try_rec = None
update_step_counter_rec = None



''' System parameters '''                                                                          
# Robot upload data
URDF_FILENAME = "ur5_robot.urdf" 
modelPath = os.getcwd()+"/urdf/" + URDF_FILENAME  
robot = RobotWrapper.BuildFromURDF(modelPath, [modelPath])
nq = robot.nq
nv = robot.nv
nx = nq + nv
na = robot.na
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()
end_effector_frame_id = 'EE'

# Dynamics parameters'
simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(robot.na)                                                                       # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'
use_viewer = False
simulate_real_time = True
show_floor = False
PRINT_T = 1                   # print every PRINT_N time steps
DISPLAY_T = 0.02              # update robot configuration in viewer every DISPLAY_N time steps
# CAMERA_TRANSFORM = [3.6914889812469482, 0.4583563506603241, -0.05435386672616005, 0.48037904500961304, 0.5339481830596924, 0.5137122273445129, 0.4692920446395874]
CAMERA_TRANSFORM = [0.36461642384529114, 0.8866147994995117, 2.579286575317383, 0.03584412857890129, 0.14833784103393555, 0.9481053948402405, 0.27893954515457153]

q_init, v_init = np.array([0.,-math.pi/2,0.,0.,0.,0.]), np.zeros(robot.nv)
simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max, use_viewer, DISPLAY_T, CAMERA_TRANSFORM, show_floor)

# State parameters 
dt = 0.01                                                                                                   # Timestep   
nb_state = robot.nq + robot.nv + 1                                                                          # State size (robot state size +1)
x_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0])                                           # State lower bound vector
x_init_min = np.array([-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi/4, -math.pi/4, -math.pi/4, -math.pi/4, -math.pi/4, -math.pi/4, 0])              # State lower bound initial configuration vector
x_max = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf, np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])                               # State upper bound vector
x_init_max = np.array([ math.pi,  math.pi,  math.pi,  math.pi,  math.pi,  math.pi, math.pi/4,  math.pi/4,  math.pi/4, math.pi/4,  math.pi/4,  math.pi/4, (NSTEPS-1)*dt])    # State upper bound initial configuration vector
state_norm_arr = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, int(NSTEPS*dt)])                                               # Array used to normalize states

# initial configurations for plot.rollout()
init_states_sim = [np.array([math.pi/4,    -math.pi/8, -math.pi/8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),                             
                   np.array([-math.pi/4,   math.pi/8,  math.pi/8,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([math.pi/2,    0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([-math.pi/2,   0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([3*math.pi/4,  0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([-3*math.pi/4, 0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([math.pi/4,    0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([-math.pi/4,   0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([math.pi,      0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

# Action parameters
nb_action = robot.na                                                                                        # Action size
u_min = np.array([-150, -150, -150, -28, -28, -28])                                                                  # Action lower bound vector
u_max = np.array([150, 150, 150, 28, 28, 28])                                                                 # Action upper bound vector
w_b = 1/w_u


fig_ax_lim = np.array([[-3, 3], [-3, 3]])                                                               # Figure axis limit [x_min, x_max, y_min, y_max]


profile = 0

env_RL = 0
