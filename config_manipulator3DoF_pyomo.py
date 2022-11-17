import numpy as np

ep_no_update = 100                       # Episodes to wait before starting to update the NNs
NEPISODES = 50000+ep_no_update            # Max training episodes
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
'''
Prioritized replay introduces bias because it changes the sample distribution in an uncontrolled fashion (while the expectation is estimated with the mean
so it would need a uniform sample distribution), and therefore changes the solution that the estimates will converge to. We can correct this bias by using importance-sampling weights: w_i = (1 / (N*P(i)) )**β
that fully compensates for the non-uniform probabilities P(i) if β = 1. The unbiased nature of the updates is most important near convergence at the end of training, as the process is highly non-stationary anyway,
due to changing policies, state distributions and bootstrap targets; so a small bias can be ignored in this context.
'''
prioritized_replay_beta0 = 0.6          
prioritized_replay_beta_iters = None    # Therefore let's exploit the flexibility of annealing the amount of IS correction over time, by defining a schedule on the exponent β
                                        # that from its initial value β0 reaches 1 only at the end of learning.
prioritized_replay_eps = 1e-5           # It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero

NORMALIZE_INPUTS = 1                    # Flag to normalize inputs (state and action)

num_states = 7                          # Number of states
num_actions = 3                         # Number of actions
tau_upper_bound = 200                   # Action upper bound
tau_lower_bound = -200                  # Action upper bound
state_norm_arr = np.array([15,15,15,10,
                10,10,int(NSTEPS*dt)])  # Array used to normalize states

EPISODE_CRITIC_PRETRAINING = 0          # Episodes of critic pretraining
EPISODE_ICS_INIT = 0                    # Episodes where ICS warm-starting is used instead of actor rollout

wreg_l1_A = 1e-2                        # Weight of L1 regularization in actor's network
wreg_l2_A = 1e-2                        # Weight of L2 regularization in actor's network
wreg_l1_C = 1e-2                        # Weight of L1 regularization in critic's network
wreg_l2_C = 1e-2                        # Weight of L2 regularization in critic's network

TD_N = 1                                # Flag to use n-step TD rather than 1-step TD
nsteps_TD_N = 1                         # Number of lookahed steps

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

w_d = 100                               # Distance from target weight
w_v = 10000                             # Velocity weight
w_u = 1                                 # Control effort weight 
w_peak = 500000                         # Target threshold weight    
w_ob1 = 5000000                         # Ellipse 1 weight
w_ob2 = 5000000                         # Ellipse 2 weight            
w_ob3 = 5000000                         # Ellipse 3 weight
alpha = 50                              # Soft abs coefficient (obstacle)
alpha2 = 50                             # Soft abs coefficient (peak)                             

M = 0.5                                 # Link mass
l = 10                                  # Link length
Iz = (M*l**2)/3                         # Inertia around z axis passing though an end of a link of mass M and length l
x_base = -7.0                           # x coord base
y_base = 0.0                            # y coord base
x_des = -20.0                           # Target x position
y_des = 0.0                             # Target y position
TARGET_STATE = np.array([x_des,y_des])  # Target state


'''' Needed in dynamics_manipulator3DoF '''
simulate_coulomb_friction = 0           # To simulate friction
simulation_type = 'euler'               # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(3)          # Expressed as percentage of torque max
