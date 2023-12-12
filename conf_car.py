import math
import numpy as np

system_id = 'car'

''' CACTO parameters '''
EP_UPDATE = 250                                                                                             # Number of episodes before updating critic and actor
NUPDATES = 260000                                                                                           # Max NNs updates
UPDATE_LOOPS = np.arange(1000, 38000, 3000)                                                                 # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS))                                                                # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                                  # Number of algorithm loops
NSTEPS = 500                                                                                                # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                                                 # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                                  # Learning rate for the policy network
REPLAY_SIZE = 2**16                                                                                         # Size of the replay buffer
BATCH_SIZE = 64                                                                                             # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                                                                      # Flag to use MC or TD(n)
if not MC:
    UPDATE_RATE = 0.001                                                                                     # Homotopy rate to update the target critic network if TD(n) is used
    nsteps_TD_N = int(NSTEPS/4)                                                                             # Number of lookahed steps if TD(n) is used


### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  10000                                                                                  # Save NNs interval
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
w_d = 1e2                                                                                                   # Distance from target weight
w_u = 1e1                                                                                                    # Control effort weight
w_peak = 5e5                                                                                                # Target threshold weight
w_ob = 5e6                                                                                                  # Obstacle weight
w_v = 0                                                                                                     # Velocity weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])                                                            # Weights vector (tmp)
cost_weights_running  = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, w_u])                                  # Running cost weights vector
cost_weights_terminal = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, 0])                                    # Terminal cost weights vector 

### SoftMax parameters 
alpha = 50                                                                                                  # Soft abs coefficient (obstacle) 
alpha2 = 5                                                                                                  # Soft abs coefficient (peak)
soft_max_param = np.array([alpha, alpha2])                                                                  # Soft parameters vector

### Cost function parameters
offset_cost_fun = 0                                                                                         # Reward/cost offset factor
scale_cost_fun = 1e-5                                                                                       # Reward/cost scale factor (1e-5)                                                                       
cost_funct_param = np.array([offset_cost_fun, scale_cost_fun])

### Target parameters
x_des = -7.0                                                                                                # Target x position
y_des = 0.0                                                                                                 # Target y position
TARGET_STATE = np.array([x_des,y_des])                                                                      # Target position



''' Path parameters '''
test_set = 'set test'                                                                                       # Test id  
Config_path = './Results Car/Results {}/Configs/'.format(test_set)                                          # Configuration path
Fig_path = './Results Car/Results {}/Figures'.format(test_set)                                              # Figure path
NNs_path = './Results Car/Results {}/NNs'.format(test_set)                                                  # NNs path
Log_path = './Results Car/Results {}/Log/'.format(test_set)                                                 # Log path
Code_path = './Results Car/Results {}/Code/'.format(test_set)                                               # Code path
DictWS_path = './Results Car/Results {}/DictWS/'.format(test_set)                                           # DictWS path
path_list = [Fig_path, NNs_path, Log_path, Code_path, DictWS_path]                                          # Path list

# Recover-training parameters
test_set_rec = None
NNs_path_rec = './Results Car/Results set {}/NNs'.format(test_set_rec)                                      # NNs path recover training
N_try_rec = None
update_step_counter_rec = None

''' System parameters ''' 
env_RL = 0                                                                                                  # Flag RL environment: set True if RL_env and TO_env are different                                                                                 

### Dynamics parameters
dt = 0.05                                                                                                   # Timestep   

simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(2)                                                                              # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'                                                                             # Link mass

### State parameters 
nb_state = 5 + 1                                                                                            # State size (robot state size +1)
nq = None
nv = None
nx = 5
na = 2
x_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0])                                          # State lower bound vector
x_init_min = np.array([-15, -15, -math.pi, -10, -3, 0])                                                     # State lower bound initial configuration array
x_max = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])                                          # State upper bound vector
x_init_max = np.array([ 15,  15,  math.pi, 10, 3, (NSTEPS-1)*dt])                                           # State upper bound initial configuration array
state_norm_arr = np.array([15, 15, math.pi, 10, 3, int(NSTEPS*dt)])                                         # Array used to normalize states
# state: x, y, theta, v, a, t

# initial configurations for plot.rollout()
init_states_sim = [np.array([2.0,   0.0,   0.0, 0.0, 0.0, 0.0]),
                   np.array([10.0,  0.0,   0.0, 0.0, 0.0, 0.0]),
                   np.array([10.0,  -10.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([10.0,  10.0,  0.0, 0.0, 0.0, 0.0]),
                   np.array([-10.0, 10.0,  0.0, 0.0, 0.0, 0.0]),
                   np.array([-10.0, -10.0, 0.0, 0.0, 0.0, 0.0]),
                   np.array([12.0,  2.0,   0.0, 0.0, 0.0, 0.0]),
                   np.array([12.0,  -2.0,  0.0, 0.0, 0.0, 0.0]),
                   np.array([15.0,  0.0,   0.0, 0.0, 0.0, 0.0])]

### Action parameters
nb_action = 2                                                                                               # Action size
bound_actions = 0
omega_lower_bound = -2                                                                                      # Action lower bound
omega_upper_bound = 2     
jerk_lower_bound = -1
jerk_upper_bound = 1                                                                                        # Action upper bound
u_min = np.array([omega_lower_bound, jerk_lower_bound])                                                     # Action lower bound vector
u_max = np.array([omega_upper_bound, jerk_upper_bound])                                                     # Action upper bound vector
w_b = 1/w_u



### Plot parameters
fig_ax_lim = np.array([[-16, 16], [-16, 16]])                                                               # Figure axis limit [x_min, x_max, y_min, y_max]



profile = 0                                                                                                 # Profile flag