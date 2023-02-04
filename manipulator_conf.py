import numpy as np
import math
from robot_utils import RobotWrapper, RobotSimulator
import os

''' Useful parameters ''' 
seed = 123



''' CACTO parameters '''
ep_no_update = 100                      # Episodes to wait before starting to update the NNs
NEPISODES = 50000+ep_no_update          # Max training episodes
EP_UPDATE = 25                          # Number of episodes before updating critic and actor
NSTEPS = 100                            # Max episode length
CRITIC_LEARNING_RATE = 0.001            # Learning rate for the critic network
ACTOR_LEARNING_RATE = 0.0005            # Learning rate for the policy network
UPDATE_RATE = 0.0005                    # Homotopy rate to update the target critic network
UPDATE_LOOPS = 160                      # Number of updates of both critic and actor performed every EP_UPDATE episodes  
REPLAY_SIZE = 2**15                     # Size of the replay buffer
BATCH_SIZE = 64                         # Size of the mini-batch 
NH1 = 256                               # 1st hidden layer size
NH2 = 256                               # 2nd hidden layer size
dt = 0.05                               # Timestep     

log_rollout_interval = 10
log_interval = 50

LR_SCHEDULE = 1                         # Flag to use a scheduler for the learning rates

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

prioritized_replay_alpha = 0            # α determines how much prioritization is used, set to 0 to use a normal buffer.
                                        # Used to define the probability of sampling transition i --> P(i) = p_i**α / sum(p_k**α) where p_i is the priority of transition i 

# Prioritized replay introduces bias because it changes the sample distribution in an uncontrolled fashion (while the expectation is estimated with the mean
# so it would need a uniform sample distribution), and therefore changes the solution that the estimates will converge to. We can correct this bias by using importance-sampling weights: w_i = (1 / (N*P(i)) )**β
# that fully compensates for the non-uniform probabilities P(i) if β = 1. The unbiased nature of the updates is most important near convergence at the end of training, as the process is highly non-stationary anyway,
# due to changing policies, state distributions and bootstrap targets; so a small bias can be ignored in this context.
prioritized_replay_beta0 = 0.6          
prioritized_replay_beta_iters = None    # Therefore let's exploit the flexibility of annealing the amount of IS correction over time, by defining a schedule on the exponent β
                                        # that from its initial value β0 reaches 1 only at the end of learning.
prioritized_replay_eps = 1e-5           # It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero

NORMALIZE_INPUTS = 1                    # Flag to normalize inputs (state and action)

EPISODE_CRITIC_PRETRAINING = 0          # Episodes of critic pretraining
EPISODE_ICS_INIT = 0                    # Episodes where ICS warm-starting is used instead of actor rollout

wreg_l1_A = 1e-2                        # Weight of L1 regularization in actor's network
wreg_l2_A = 1e-2                        # Weight of L2 regularization in actor's network
wreg_l1_C = 1e-2                        # Weight of L1 regularization in critic's network
wreg_l2_C = 1e-2                        # Weight of L2 regularization in critic's network

TD_N = 1                                # Flag to use n-step TD rather than 1-step TD
nsteps_TD_N = 1                         # Number of lookahed steps



''' Recover training parameters'''
recover_stopped_training = 0
update_step_counter = 0 #102400 



''' Cost function parameters '''
# Obstacles parameters
XC1 = -2.0                              # X coord center ellipse 1
YC1 = 0.0                               # Y coord center ellipse 1
A1 = 6                                  # Width ellipse 1 
B1 = 10                                 # Height ellipse 1 
XC2 = 3.0                               # X coord center ellipse 2 
YC2 = 4.0                               # Y coord center ellipse 2
A2 = 12                                 # Width ellipse 2 
B2 = 4                                  # Height ellipse 2 
XC3 = 3.0                               # X coord center ellipse 2 
YC3 = -4.0                              # Y coord center ellipse 2
A3 = 12                                 # Width ellipse 2 
B3 = 4                                  # Height ellipse 2 
obs_param = np.array([XC1, YC1, XC2, YC2, XC3, YC3, A1, B1, A2, B2, A3, B3])

# Weigths
w_d = 100                              # Distance from target weight
w_u = 1                                # Control effort weight
w_peak = 5e5                           # Target threshold weight
w_ob = 5e6                             # Obstacle weight
w_v = 1e4                              # Velocity weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])

# SoftMax parameters 
alpha = 50                            # Soft abs coefficient (obstacle) 
alpha2 = 50                           # Soft abs coefficient (peak)
soft_max_param = np.array([alpha, alpha2])

# Target parameters
x_des = -20.0                           # Target x position
y_des = 0.0                             # Target y position
TARGET_STATE = np.array([x_des,y_des])  # Target state



''' Path parameters '''
#check = 0
N_try = 1                                                                   # Number of test run
Fig_path = './Results/Figures/N_try_{}'.format(N_try)
NNs_path = './Results/NNs/N_try_{}'.format(N_try)
Config_path = './Results/Configs/'
Test_path = './Results/Test/'



''' System parameters '''
# Dynamics parameters'
simulate_coulomb_friction = 0           # To simulate friction
simulation_type = 'euler'               # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(3)          # Expressed as percentage of torque max

# Robot upload data
URDF_FILENAME = "planar_manipulator_3dof.urdf" 
modelPath = os.getcwd()+"/urdf/" + URDF_FILENAME  #os.getcwd()+"/../description_file/urdf/" + URDF_FILENAME 
robot = RobotWrapper.BuildFromURDF(modelPath, [modelPath])
q_init, v_init = np.array([math.pi, math.pi, math.pi]), np.zeros(robot.nv)
simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max)

# System configuration parameters
M = 0.5                                 # Link mass
l = 10                                  # Link length
Iz = (M*l**2)/3                         # Inertia around z axis passing though an end of a link of mass M and length l
x_base = -7.0                           # x coord base
y_base = 0.0                            # y coord base

# State parameters 
x_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0])                  #not used
x_init_min = np.array([-math.pi, -math.pi, -math.pi, -math.pi/4, -math.pi/4, -math.pi/4, 0]) #not used
x_max = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf])             #not used
x_init_max = np.array([ math.pi,  math.pi,  math.pi,  math.pi/4,  math.pi/4,  math.pi/4, (NSTEPS-1)*dt]) #not used
nb_state = robot.nq + robot.nv + 1
state_norm_arr = np.array([15,15,15,10,10,10,int(NSTEPS*dt)])  # Array used to normalize states

# Action parameters
nb_action = robot.na
tau_upper_bound = 200                       # Action upper bound
tau_lower_bound = -200                      # Action upper bound
u_min = tau_lower_bound*np.ones(nb_action)  #not used
u_max = tau_upper_bound*np.ones(nb_action)  #not used
