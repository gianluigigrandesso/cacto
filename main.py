import os
import sys
import time
import math
import pickle
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
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                         type=int,   default=0,                                    
                        help="Test number")
    
    parser.add_argument('--seed',                           type=int,   default=None,                                    
                        help="random and tf.random seed")

    parser.add_argument('--system-id',                      type=str,   default='double_integrator',
                        choices=["double_integrator", "manipulator"],
                        help="System-id, either double_integrator or manipulator")
    
    parser.add_argument('--TO-method',                      type=str,   default='casadi',
                        choices=["pyomo", "casadi"],
                        help="Method to solve TO problem, either pyomo or casadi")

    parser.add_argument('--recover-training-flag',          type=bool,  default=False,
                        choices=["True", "False"],
                        help="Flag to recover training")
    
    parser.add_argument('--CPU-flag',                       type=bool,  default=False,
                        choices=["True", "False"],
                        help="Flag to use CPU")
    
    parser.add_argument('--nb-cpus',                        type=int,   default=1,
                        help="Number of TO problems solved in parallel")
    
    parser.add_argument('--w-S',                            type=float, default=0,
                        help="Sobolev training weight")
    
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args




if __name__ == '__main__':

    args = parse_args()
    
    ###           Input           ###
    N_try     = args['test_n']

    if args['seed'] == None:
        seed = random.randint(1,100000)
    else:
        seed = args['seed']
    tf.random.set_seed(seed)  # Set tensorflow seed
    random.seed(seed)         # Set random seed

    system_id = args['system_id'] 

    TO_method = args['TO_method']

    recover_training_flag = args['recover_training_flag']
    
    CPU_flag = args['CPU_flag'] 
    if CPU_flag:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    tf.config.experimental.list_physical_devices('GPU')
    
    nb_cpus = args['nb_cpus']

    w_S = args['w_S']
    #################################



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



    # Create folders to store the results and the trained NNs and save configuration
    for path in conf.path_list:
        try:
            os.makedirs(path + '/N_try_{}'.format(N_try))                                                  
        except:
            print("N try = {} {} folder already existing".format(N_try, path))
            pass
    try:
        os.makedirs(conf.Config_path)                                                  
    except:
        print("{} folder already existing".format(conf.Config_path))
        pass

    params = [n for n in conf.__dict__ if not n.startswith("__")]
    f = open(conf.Config_path+'/config{}.txt'.format(N_try), 'w') #
    for p in params:
        f.write('{} = {}\n'.format(p, conf.__dict__[p]))
    f.write('Seed = {}\n'.format(seed))
    f.write('w_S = {}'.format(w_S))
    f.close()



    # Create environment instance
    env = Environment(conf)

    # Create NN instance
    NN_inst = NN(env, conf, w_S)

    # Create RL_AC instance 
    RLAC = RL_AC(env, NN_inst, conf, N_try)

    # Set initial weights of the NNs, initialize the counter of the updates and setup NN models
    if recover_training_flag:
        recover_training = np.array([conf.NNs_path_rec, conf.N_try_rec, conf.update_step_counter_rec])
        update_step_counter = conf.update_step_counter_rec
        nb_starting_episode = (conf.update_step_counter_rec/conf.UPDATE_LOOPS)+1

        RLAC.setup_model(recover_training)
    else:
        update_step_counter = 0
        nb_starting_episode = 0

        RLAC.setup_model()

    RLAC.RL_save_weights(update_step_counter)

    # Select TO method and create TO instance
    if TO_method == 'pyomo':
        TrOp = TO_Pyomo(env, conf, system_id, w_S)
    elif TO_method == 'casadi':
        TrOp = TO_Casadi(env, conf, system_id, w_S)
    else:
        print('TO method: ' + TO_method + ' not implemented')
        sys.exit()

    # Create PLOT instance
    plot_fun = PLOT(N_try, env, conf)

    # Create an empty (prioritized) replay buffer
    if conf.prioritized_replay_alpha == 0:
        buffer = ReplayBuffer(conf)   
    else:
        buffer = PrioritizedReplayBuffer(conf)   



    def run_parallel(n=1):

        success_flag = 0                # Flag to indicate if the TO problem has been solved
        while success_flag == 0:

            # Create initial TO #
            success_init_flag = 0
            while success_init_flag == 0:
                init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH, success_init_flag = RLAC.create_TO_init()
            
            # Solve TO problem #
            TO_controls, TO_states, success_flag, ee_pos_arr_TO, dVdx = TrOp.TO_Solve(init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
        
        # Collect experiences 
        state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL  = RLAC.RL_Solve(TO_controls, TO_states)

        if conf.env_RL == 0:
            ee_pos_arr_RL = ee_pos_arr_TO

        return NSTEPS_SH, TO_controls, success_flag, ee_pos_arr_TO, dVdx, state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL



    # Arrays to store the reward history of each episode and the average reward history of last few episodes
    ep_reward_arr = np.empty(conf.NEPISODES-nb_starting_episode)                                                                                     
    avg_reward_arr = np.empty(conf.NLOOPS-nb_starting_episode)  



    ### START TRAINING ###
    if conf.profile:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    time_start = time.time()

    for ep in range(nb_starting_episode, conf.NLOOPS): 

        # Solve TO problem and collect experiences
        if nb_cpus > 1:
            with Pool(nb_cpus) as p: 
                tmp = p.map(run_parallel,range(conf.EP_UPDATE))
            NSTEPS_SH, TO_controls, success_flag, ee_pos_arr_TO, dVdx, state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL = zip(*tmp)

            for j in range(conf.EP_UPDATE):
                # Update the buffer
                [buffer.add(state_arr[j][i], partial_cost_to_go_arr[j][i], state_next_rollout_arr[j][i], dVdx[j][i], done_arr[j][i], term_arr[j][i]) for i in range(len(rwrd_arr[j]))]

        else:
            ep_return = np.empty(conf.EP_UPDATE)
            for j in range(conf.EP_UPDATE):
                NSTEPS_SH, TO_controls, success_flag, ee_pos_arr_TO, dVdx, state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return[j], ee_pos_arr_RL = run_parallel()

                # Update the buffer
                [buffer.add(state_arr[i], partial_cost_to_go_arr[i], state_next_rollout_arr[i], dVdx[i], done_arr[i], term_arr[i]) for i in range(len(rwrd_arr))]

        # Update NNs
        update_step_counter = RLAC.learn_and_update(update_step_counter, buffer, ep)

        # Plot rollouts and state and control trajectories
        if update_step_counter%conf.plot_rollout_interval_diff_loc == 0:
            _ = plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim, diff_loc=1)

        ep_reward_arr[ep*conf.EP_UPDATE:(ep+1)*conf.EP_UPDATE] = ep_return
        avg_reward = np.mean(ep_reward_arr[-40:])  # Mean of last 40 episodes
        avg_reward_arr[ep] = avg_reward

        for i in range(conf.EP_UPDATE):
            print("Episode  {}  --->   Return = {}".format(ep*conf.EP_UPDATE + i, ep_return[i]))

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    if conf.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

    # Plot returns
    plot_fun.plot_AvgReturn(avg_reward_arr)
    plot_fun.plot_Return(ep_reward_arr)

    # Save networks at the end of the training
    RLAC.RL_save_weights()

    # Simulate the final policy
    plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim)
