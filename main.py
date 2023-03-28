import os
import sys
import time
import math
import random
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import tensorflow as tf
import matplotlib.pyplot as plt
from pyomo.dae import *
from pyomo.environ import *
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from RL import RL_AC 
from plot import PLOT
from TO import TO_Pyomo, TO_Casadi


def run(**kwargs):

    ###           Input           ###
    N_try     = kwargs['test_n']
    system_id = kwargs['system_id'] 
    TO_method = kwargs['TO_method']
    if kwargs['seed'] == None:
        seed = random.randint(1,100000)
        print(seed)
    else:
        seed = kwargs['seed']
    CPU_flag = kwargs['CPU_flag'] 
    if CPU_flag:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" # To run TF on CPU rather than GPU 
        #(seems faster since the NNs are small and some gradients are computed with 
        # Pinocchio on CPU --> bottleneck = communication CPU-GPU?)
    tf.config.experimental.list_physical_devices('GPU')
    ##################################



    if system_id == 'manipulator':
        import manipulator_conf as conf
        from environment import Manipulator as Environment

    elif system_id == 'double_integrator':
        import double_integrator_conf as conf
        from environment import DoubleIntegrator as Environment

    else:
        print('System {} not found'.format(system_id))
        sys.exit()

    # Create folders to store the results and the trained NNs
    try:
        os.makedirs(conf.Fig_path + '/N_try_{}'.format(N_try))                                                  
    except:
        print("N try = {} Figures folder already existing".format(N_try))
        pass
    try:
        os.makedirs(conf.Fig_path + '/N_try_{}'.format(N_try) +'/Actor')                                         
    except:
        print("N try = {} Actor folder already existing".format(N_try))
        pass
    try:
        os.makedirs(conf.NNs_path + '/N_try_{}'.format(N_try))                                                 
    except:
        print("N try = {} NNs folder already existing".format(N_try))
        pass
    try:
        os.makedirs(conf.Config_path)                                              
    except:
        print("N try = {} Configs folder already existing".format(N_try))
        pass

    # Save config file
    params = [n for n in conf.__dict__ if not n.startswith("__")]
    f = open(conf.Config_path+'/config{}.txt'.format(N_try), 'w')
    for p in params:
        f.write('{} = {}\n'.format(p, conf.__dict__[p]))
    f.close()

    # Set seed
    tf.random.set_seed(seed)  
    random.seed(seed)

    # Create environment 
    env = Environment(conf)

    # Create RL_AC instance 
    RLAC = RL_AC(env, conf)
    RLAC.setup_model()

    # Select TO method and create TO instance
    if TO_method == 'pyomo':
        TrOp = TO_Pyomo(env, conf, system_id)
    elif TO_method == 'casadi':
        TrOp = TO_Casadi(env, conf, system_id)
    else:
        print('TO method: ' + TO_method + ' not implemented')
        sys.exit()

    # Create PLOT instance
    plot_fun = PLOT(N_try, env, conf)

    # Set initial weights of the NNs and initialize the counter of the updates
    if conf.recover_stopped_training:
        nb_starting_episode = ((conf.update_step_counter/conf.UPDATE_LOOPS)*conf.EP_UPDATE)+1
    else: 
        nb_starting_episode = 0

    # Create an empty (prioritized) replay buffer
    if conf.prioritized_replay_alpha != 0:
        buffer = ReplayBuffer(conf.REPLAY_SIZE, alpha=conf.prioritized_replay_alpha)   
    else:
        buffer = PrioritizedReplayBuffer(conf.REPLAY_SIZE, alpha=conf.prioritized_replay_alpha)   

    # Lists to store the reward history of each episode and the average reward history of last few episodes
    ep_reward_list = np.empty(conf.NEPISODES-nb_starting_episode)                                                                                     
    avg_reward_list = np.empty(conf.NEPISODES-nb_starting_episode)   

    time_start = time.time()

    ### START TRAINING ###
    for ep in range(nb_starting_episode,conf.NEPISODES): 

        # START TO PROBLEM #
        success_flag = 0             # Flag to indicate if the TO problem has been solved
        while success_flag==0:
            # Create initial TO #
            init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH = RLAC.create_TO_init()
            
            # Solve TO problem #
            tau_TO, success_flag, x_ee_arr_TO, y_ee_arr_TO = TrOp.TO_Solve(ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)

        # START RL PROBLEM # 
        ep_return, update_step_counter, x_ee_arr_RL, y_ee_arr_RL = RLAC.RL_Solve(ep, tau_TO, buffer)
        
        # Plot the state and control trajectories of this episode
        # if ep >= conf.ep_no_update and ep%conf.log_rollout_interval==0:
        #     plot_fun.plot_results(tau_TO, x_ee_arr_TO, y_ee_arr_TO, x_ee_arr_RL, y_ee_arr_RL, NSTEPS_SH, to=success_flag)

        # Plot rollouts every conf.log_rollout_interval-training episodes (saved in a separate folder)
        # if ep >= conf.ep_no_update and ep%conf.log_rollout_interval==0:
        #     _, _, _ = plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim, diff_loc=1)     

        # Plot rollouts and save the NNs every conf.log_rollout_interval-training episodes
        # if ep >= conf.ep_no_update and ep%conf.log_interval==0: 
        #     _, _, _ = plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim)        
        #     RLAC.actor_model.save_weights(conf.NNs_path+"/actor_{}.h5".format(update_step_counter))
        #     RLAC.critic_model.save_weights(conf.NNs_path+"/critic_{}.h5".format(update_step_counter))
        #     RLAC.target_critic.save_weights(conf.NNs_path+"/target_critic_{}.h5".format(update_step_counter))
 
        ep_reward_list[ep] = ep_return
        avg_reward = np.mean(ep_reward_list[-40:])  # Mean of last 40 episodes
        avg_reward_list[ep] = avg_reward

        print("Episode  {}  --->   Return = {}".format(ep, ep_return))

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    # Plot returns
    plot_fun.plot_AvgReturn(avg_reward_list)
    plot_fun.plot_Return(ep_reward_list)

    # Save networks at the end of the training
    RLAC.actor_model.save_weights(conf.NNs_path+"/actor_final.h5")
    RLAC.critic_model.save_weights(conf.NNs_path+"/critic_final.h5")
    RLAC.target_critic.save_weights(conf.NNs_path+"/target_critic_final.h5")

    # Simulate the final policy
    tau_all_final_sim, x_ee_all_final_sim, y_ee_all_final_sim = plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim)

def parse_args():
    """
    parse the arguments for CACTO training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                         type=int,   default=0)
    parser.add_argument('--system-id',                      type=str,   default='-')
    parser.add_argument('--TO-method',                      type=str,   default='pyomo')
    parser.add_argument('--seed',                           type=int,   default=None)
    parser.add_argument('--CPU-flag',                       type=bool,  default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args




if __name__ == '__main__':
    args = parse_args()
    run(**args)
