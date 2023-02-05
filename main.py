import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
from replay_buffer import PrioritizedReplayBuffer
from plot import PLOT
from CACTO import CACTO
from TO import TO_Pyomo, TO_Casadi
from RL import RL_AC 
import numpy as np
import random
import time
import math

def run():

    time_start = time.time()

    ### Select system and TO method ###
    system = 'double_integrator'
    #system = 'manipulator'
    TO_method = 'pyomo'
    seed = 123
    ##################################


    if system == 'manipulator':
        import manipulator_conf as conf
        from environment import Manipulator as Environment

        init_states_sim = [np.array([math.pi/4,-math.pi/8,-math.pi/8,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,math.pi/8,math.pi/8,0.0,0.0,0.0,0.0]),np.array([math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi,0.0,0.0,0.0,0.0,0.0,0.0])]

    elif system == 'double_integrator':
        import double_integrator_conf as conf
        from environment import DoubleIntegrator as Environment

        init_states_sim = [np.array([2.0,0.0,0.0,0.0,0.0]),np.array([10.0,0.0,0.0,0.0,0.0]),np.array([10.0,-10.0,0.0,0.0,0.0]),np.array([10.0,10.0,0.0,0.0,0.0]),np.array([-10.0,10.0,0.0,0.0,0.0]),np.array([-10.0,-10.0,0.0,0.0,0.0]),np.array([12.0,2.0,0.0,0.0,0.0]),np.array([12.0,-2.0,0.0,0.0,0.0]),np.array([15.0,0.0,0.0,0.0,0.0])]

    else:
        print('System: ' + system + ' not found')


    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Uncomment to run TF on CPU rather than GPU                         

    # To run TF on CPU rather than GPU (seems faster since the NNs are small and some gradients are computed with Pinocchio on CPU --> bottleneck = communication CPU-GPU?)
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"     
    tf.config.experimental.list_physical_devices('GPU')

    # Create folders to store the results and the trained NNs
    try:
        os.makedirs(conf.Fig_path)                                                  
    except:
        print("N try = {} Figures folder already existing".format(conf.N_try))
        pass
    try:
        os.makedirs(conf.Fig_path+'/Actor')                                         
    except:
        print("N try = {} Actor folder already existing".format(conf.N_try))
        pass
    try:
        os.makedirs(conf.NNs_path)                                                 
    except:
        print("N try = {} NNs folder already existing".format(conf.N_try))
        pass
    try:
        os.makedirs(conf.Config_path)                                              
    except:
        print("N try = {} Configs folder already existing".format(conf.N_try))
        pass

    # Save config file
    f=open(conf.Config_path+'/'+system+'_config{}.txt'.format(conf.N_try), 'w')
    f.write("conf.NEPISODES = {}, conf.NSTEPS = {}, conf.CRITIC_LEARNING_RATE = {}, conf.ACTOR_LEARNING_RATE = {}, conf.UPDATE_RATE = {}, conf.REPLAY_SIZE = {}, conf.BATCH_SIZE = {}, conf.NH1 = {}, conf.NH2 = {}, conf.dt = {}".format(conf.NEPISODES,conf.NSTEPS,conf.CRITIC_LEARNING_RATE,conf.ACTOR_LEARNING_RATE,conf.UPDATE_RATE,conf.REPLAY_SIZE,conf.BATCH_SIZE,conf.NH1,conf.NH2,conf.dt)+
            "\n"+str(conf.UPDATE_LOOPS)+" updates every {} episodes".format(conf.EP_UPDATE)+
            "\n\nReward = ({}*(-(x_ee-conf.TARGET_STATE[0])**2 -(y_ee-conf.TARGET_STATE[1])**2) + {}*peak_reward - {}*vel_joint - {}*ell1_pen - {}*ell2_pen - {}*ell3_pen - {}*(u[0]**2 + u[1]**2 + u[2]**2))/100, vel_joint = x2[3]**2 + x2[4]**2 + x2[5]**2 - 10000/{} if final step else 0, peak reward = math.log(math.exp({}*-(x_err-0.1 + y_err-0.1)) + 1)/{}, x_err = math.sqrt((x_ee-conf.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1), y_err = math.sqrt((y_ee-conf.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1), ell_pen = log(exp({}*-(((x_ee-XC)**2)/((a/2)**2) + ((y_ee-YC)**2)/((conf.B1/2)**2) - 1.0)) + 1)/{}".format(conf.w_d,conf.w_peak,conf.w_v,conf.w_ob,conf.w_ob,conf.w_ob,conf.w_u,conf.w_v,conf.alpha2,conf.alpha2,conf.alpha,conf.alpha)+
            "\n\ntarget (conf.TARGET_STATE) = "+str(conf.TARGET_STATE)+
            "\nPrioritized_replay_alpha = "+str(conf.prioritized_replay_alpha)+", conf.prioritized_replay_beta0 = "+str(conf.prioritized_replay_beta0)+", conf.prioritized_replay_eps = "+str(conf.prioritized_replay_eps)+
            "\nActor: kernel_and_bias_regularizer = l1_l2({}), Critic:  kernel_and_bias_regularizer = l1_l2({}) (in each layer)".format(conf.wreg_l1_A,conf.wreg_l2_A,conf.wreg_l1_C,conf.wreg_l2_C)+
            "\nScheduled step LR decay = {}: critic values = {} and boundaries = {}, policy values = {} and boundaries = {}".format(conf.LR_SCHEDULE,conf.values_schedule_LR_C,conf.boundaries_schedule_LR_C,conf.values_schedule_LR_A,conf.boundaries_schedule_LR_A)+
            "\nRandom initial state -> [uniform(-pi,pi), uniform(-pi,pi), uniform(-pi,pi), uniform(-pi/4,pi/4), uniform(-pi/4,pi/4), uniform(-pi/4,pi/4),uniform(0,(NSTEPS_SH-1)*conf.dt)"+ 
            "\nNormalized inputs = {}, q by {} and qdot by {}".format(conf.NORMALIZE_INPUTS,conf.state_norm_arr[0],conf.state_norm_arr[3])+
            "\nEpisodes of critic pretraining = {}".format(conf.EPISODE_CRITIC_PRETRAINING)+
            "\nn-step TD = {} with {} lookahead steps".format(conf.TD_N, conf.nsteps_TD_N))
    f.close()

    # Set seed
    tf.random.set_seed(seed)  
    random.seed(seed)

    # Create environment 
    env = Environment(conf.dt, conf.x_init_min, conf.x_init_max, conf.x_min, conf.x_max, conf.u_min, conf.u_max, conf.TARGET_STATE, conf.soft_max_param, conf.obs_param, conf.weight, conf.robot, conf.nb_state, conf.nb_action, conf.NORMALIZE_INPUTS, conf.state_norm_arr, conf.simu) 
    #env.seed(seed=seed)

    # Create CACTO instance
    cacto_instance = CACTO(env, conf)
    cacto_instance.setup_model() # sys_dep: ACTOR and CRITIC NNs#

    # Select TO method and create TO instance
    if TO_method == 'pyomo':
        TrOp = TO_Pyomo(env, conf)
    elif TO_method == 'casadi':
        TrOp = TO_Casadi(env, conf)
    else:
        print('TO method: ' + TO_method + ' not implemented')
        sys.exit()

    # Create RL_AC instance 
    RLAC = RL_AC(env, conf)

    # Create PLOT instance
    plot_fun = PLOT(env, conf)

    # Set initial weights of the NNs and initialize the counter of the updates
    if conf.recover_stopped_training:
        nb_starting_episode = ((conf.update_step_counter/conf.UPDATE_LOOPS)*conf.EP_UPDATE)+1
    else: 
        nb_starting_episode = 0

    # Create an empty (prioritized) replay buffer
    prioritized_buffer = PrioritizedReplayBuffer(conf.REPLAY_SIZE, alpha=conf.prioritized_replay_alpha)   

    # Lists to store the reward history of each episode and the average reward history of last few episodes
    ep_reward_list = []                                                                                     
    avg_reward_list = []

    ### START TRAINING ###
    for ep in range(nb_starting_episode,conf.NEPISODES): 

        # TO #
        rand_time, prev_state, tau_TO = TrOp.TO_Solve(ep, env)

        # RL # 
        update_step_counter, ep_return = RLAC.RL_Solve(prev_state, ep, rand_time, env, tau_TO, prioritized_buffer)

        # Plot rollouts every conf.log_rollout_interval-training episodes (saved in a separate folder)
        if ep >= conf.ep_no_update and ep%conf.log_rollout_interval==0:
            print('ciao')
            _, _, _ = plot_fun.rollout(update_step_counter, CACTO.actor_model, env, rand_time, init_states_sim, diff_loc=1)     

        # Plot rollouts and save the NNs every conf.log_rollout_interval-training episodes
        if ep >= conf.ep_no_update and conf.log_interval!=1 and conf.log_interval!=0 and ep%conf.log_interval==0: 
            _, _, _ = plot_fun.rollout(update_step_counter, CACTO.actor_model, env, rand_time, init_states_sim)        
            CACTO.actor_model.save_weights(conf.NNs_path+"/Manipulator_{}.h5".format(update_step_counter))
            CACTO.critic_model.save_weights(conf.NNs_path+"/Manipulator_critic_{}.h5".format(update_step_counter))
            CACTO.target_critic.save_weights(conf.NNs_path+"/Manipulator_target_critic_{}.h5".format(update_step_counter))

        # ep_reward_list.append(ep_return)
        # avg_reward = np.mean(ep_reward_list[-40:])  # Mean of last 40 episodes
        # avg_reward_list.append(avg_reward)

        print("Episode  {}  --->   Return = {}".format(ep, ep_return))

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    # Plot returns
    # plot_fun.plot_AvgReturn(avg_reward_list)
    # plot_fun.plot_Return(ep_reward_list)

    # Save networks at the end of the training
    CACTO.actor_model.save_weights(conf.NNs_path+"/"+system+"_actor_final.h5")
    CACTO.critic_model.save_weights(conf.NNs_path+"/"+system+"_critic_final.h5")
    CACTO.target_critic.save_weights(conf.NNs_path+"/"+system+"_target_critic_final.h5")

    # Simulate the final policy
    #tau_all_final_sim, x_ee_all_final_sim, y_ee_all_final_sim = rollout(update_step_counter)




if __name__ == '__main__':
    run()


























'''
TO-method,
system,

NORMALIZE-INPUTS, 
EPISODE-CRITIC-PRETRAINING, 
nsteps-TD-N,
NSTEPS,
EPISODE-ICS-INIT, 
TD-N, 
prioritized-replay-eps, 
prioritized-replay-alpha,
UPDATE-LOOPS, 
UPDATE-RATE, 
LR-SCHEDULE, 
update-step-counter, 
NH1, 
NH2, 
wreg-l1-A, 
wreg-l2-A, 
wreg-l1-C, 
wreg-l2-C, 
boundaries-schedule-LR-C, 
values-schedule-LR-C,
boundaries-schedule-LR-A, 
values-schedule-LR-A, 
CRITIC-LEARNING-RATE, 
ACTOR-LEARNING-RATE, 
recover-stopped-training, 
NNs-path, 
BATCH-SIZE, 

conf:
conf.tau-upper-bound, 
conf.tau-lower-bound, 
conf.dt, 
conf.soft-max-param, 
conf.obs_param,
conf.weight,
conf.TARGET-STATE,  
conf.state-norm_arr, 
conf.robot,  (conf.nb_state, conf.nb_action)
'''

'''
TO_method,
system,

conf.NORMALIZE_INPUTS, 
conf.EPISODE_CRITIC_PRETRAINING, 
conf.nsteps_TD_N,
conf.NSTEPS,
conf.EPISODE_ICS_INIT, 
conf.TD_N, 
conf.prioritized_replay_eps, 
conf.prioritized_replay_alpha,
conf.UPDATE_LOOPS, 
conf.UPDATE_RATE, 
conf.LR_SCHEDULE, 
conf.update_step_counter, 
conf.NH1, 
conf.NH2, 
conf.wreg_l1_A, 
conf.wreg_l2_A, 
conf.wreg_l1_C, 
conf.wreg_l2_C, 
conf.boundaries_schedule_LR_C, 
conf.values_schedule_LR_C,
conf.boundaries_schedule_LR_A, 
conf.values_schedule_LR_A, 
conf.CRITIC_LEARNING_RATE, 
conf.ACTOR_LEARNING_RATE, 
conf.recover_stopped_training, 
conf.NNs_path, 
conf.BATCH_SIZE, 

conf.tau_upper_bound, 
conf.tau_lower_bound, 
conf.dt, system_param,   
conf.soft_max_param, 
conf.obs_param,
conf.weight,
conf.TARGET_STATE,  
conf.state_norm_arr, 
conf.robot,  (conf.nb_state, conf.nb_action)

conf
'''