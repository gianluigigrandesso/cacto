import os
import sys
import time
import math
import random
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import tensorflow as tf
from multiprocessing import Pool
from RL import RL_AC 
from plot import PLOT
from NeuralNetwork import NN
from TO import TO_Pyomo, TO_Casadi 
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

def parse_args():
    """
    parse the arguments for CACTO training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                         type=int,   default=0)
    parser.add_argument('--system-id',                      type=str,   default='manipulator')
    parser.add_argument('--TO-method',                      type=str,   default='pyomo')
    parser.add_argument('--seed',                           type=int,   default=123)
    parser.add_argument('--CPU-flag',                       type=bool,  default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args




if __name__ == '__main__':
    args = parse_args()
    
    ###           Input           ###
    N_try     = args['test_n']
    system_id = args['system_id'] 
    TO_method = args['TO_method']
    if args['seed'] == None:
        seed = random.randint(1,100000)
        print(seed)
    else:
        seed = args['seed']
    tf.random.set_seed(seed)  # Set tensorflow seed
    random.seed(seed)         # Set random seed
    CPU_flag = args['CPU_flag'] 
    if CPU_flag:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" # To run TF on CPU rather than GPU 
        #(seems faster since the NNs are small and some gradients are computed with 
        # Pinocchio on CPU --> bottleneck = communication CPU-GPU?)
    tf.config.experimental.list_physical_devices('GPU')
    ##################################


    # Import configuration file and environment file
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



    # Create environment instance
    env = Environment(conf)

    # Create NN instance
    NN_inst = NN(env, conf)

    # Create RL_AC instance 
    RLAC = RL_AC(env, NN_inst, conf)
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
        nb_starting_episode = (conf.update_step_counter/conf.UPDATE_LOOPS)+1
        update_step_counter = conf.update_step_counter
    else: 
        nb_starting_episode = 0
        update_step_counter = 0

    # Create an empty (prioritized) replay buffer
    if conf.prioritized_replay_alpha == 0:
        buffer = ReplayBuffer(conf.REPLAY_SIZE)   
    else:
        buffer = PrioritizedReplayBuffer(conf.REPLAY_SIZE, alpha=conf.prioritized_replay_alpha, beta=conf.prioritized_replay_eps)   

    # Arrays to store the reward history of each episode and the average reward history of last few episodes
    ep_reward_arr = np.empty(conf.NEPISODES*conf.EP_UPDATE-nb_starting_episode)                                                                                     
    avg_reward_arr = np.empty(conf.NEPISODES*conf.EP_UPDATE-nb_starting_episode)   

    def run_parallel(n=1):

        success_flag = 0                # Flag to indicate if the TO problem has been solved
        while success_flag==0:

            # Create initial TO #
            init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH = RLAC.create_TO_init()
            
            # Solve TO problem #
            TO_controls, success_flag, x_ee_arr_TO, y_ee_arr_TO = TrOp.TO_Solve(ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
        
        # Collect experiences 
        state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ep_return, x_ee_arr_RL, y_ee_arr_RL  = RLAC.RL_Solve(ep, TO_controls)
        
        return NSTEPS_SH, TO_controls, success_flag, x_ee_arr_TO, y_ee_arr_TO, state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ep_return, x_ee_arr_RL, y_ee_arr_RL

    time_start = time.time()

    ### START TRAINING ###
    for ep in range(nb_starting_episode, conf.NLOOPS): 

        # Solve TO problem and collect experiences
        with Pool(conf.nb_cpus) as p: 
            tmp = p.map(run_parallel,range(conf.EP_UPDATE))
        NSTEPS_SH, TO_controls, success_flag, x_ee_arr_TO, y_ee_arr_TO, state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ep_return, x_ee_arr_RL, y_ee_arr_RL = zip(*tmp)

        # Update the buffer
        for j in range(conf.EP_UPDATE):
            for i in range(len(rwrd_arr[j])):
                buffer.add(state_arr[j][i], partial_cost_to_go_arr[j][i], state_next_rollout_arr[j][i], done_arr[j][i])
       
        # Update NNs
        update_step_counter, actor_model, critic_model, target_critic = RLAC.learn_and_update(ep, update_step_counter, buffer)
        
        # Plot rollouts and state and control trajectories
        if ep%conf.plot_rollout_interval_diff_loc==0:
            plot_fun.rollout(update_step_counter, actor_model, conf.init_states_sim, diff_loc=1)
            #plot_fun.plot_results(TO_controls[-1], x_ee_arr_TO[-1], y_ee_arr_TO[-1], x_ee_arr_RL[-1], y_ee_arr_RL[-1], NSTEPS_SH[-1], to=success_flag)
        if ep%conf.plot_rollout_interval==0: 
            plot_fun.rollout(update_step_counter, actor_model, conf.init_states_sim)         

        # Plot rollouts and save the NNs every conf.log_rollout_interval-training episodes
        if ep%conf.save_interval==0:  
            actor_model.save_weights(conf.NNs_path+"/actor_{}.h5".format(update_step_counter))
            critic_model.save_weights(conf.NNs_path+"/critic_{}.h5".format(update_step_counter))
            target_critic.save_weights(conf.NNs_path+"/target_critic_{}.h5".format(update_step_counter))

        ep_reward_arr[ep*conf.EP_UPDATE:(ep+1)*conf.EP_UPDATE] = ep_return
        avg_reward = np.mean(ep_reward_arr[-40:])  # Mean of last 40 episodes
        avg_reward_arr[ep] = avg_reward

        for i in range(conf.EP_UPDATE):
            print("Episode  {}  --->   Return = {}".format(ep*conf.EP_UPDATE + i, ep_return[i]))
        
    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    # Plot returns
    plot_fun.plot_AvgReturn(avg_reward_arr)
    plot_fun.plot_Return(ep_reward_arr)

    # Save networks at the end of the training
    actor_model.save_weights(conf.NNs_path+"/actor_final.h5")
    critic_model.save_weights(conf.NNs_path+"/critic_final.h5")
    target_critic.save_weights(conf.NNs_path+"/target_critic_final.h5")

    # Simulate the final policy
    plot_fun.rollout(update_step_counter, actor_model, conf.init_states_sim)
