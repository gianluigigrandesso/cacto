import os
import sys
import time
import shutil
import random
import argparse
import importlib
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import tensorflow as tf
from multiprocessing import Pool
from RL import RL_AC 
from TO import TO_Casadi 
from plot_utils import PLOT
from NeuralNetwork import NN
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

def parse_args():
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                         type=int,   default=0,                                    
                        help="Test number")
    
    parser.add_argument('--seed',                           type=int,   default=0,                                    
                        help="random and tf.random seed")

    parser.add_argument('--system-id',                      type=str,   default='single_integrator',
                        choices=["single_integrator", "double_integrator", "car", "car_park", "manipulator", "ur5"],
                        help="System-id (single_integrator, double_integrator, car, manipulator, ur5")

    parser.add_argument('--recover-training-flag',          type=bool,  default=False,
                        choices=["True", "False"],
                        help="Flag to recover training")
    ### Not tested ###
    #parser.add_argument('--GPU-flag',                       type=bool,  default=False,
    #                    choices=["True", "False"],
    #                    help="Flag to use GPU")
    
    parser.add_argument('--nb-cpus',                        type=int,   default=10,
                        help="Number of TO problems solved in parallel")
    
    parser.add_argument('--w-S',                            type=float, default=0,
                        help="Sobolev training - weight of the value related error")
    
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args




if __name__ == '__main__':

    args = parse_args()
    
    ###############           Input           ###############
    N_try     = args['test_n']

    if args['seed'] == None:
        seed = random.randint(1,100000)
    else:
        seed = args['seed']
    tf.random.set_seed(seed)  # Set tensorflow seed
    random.seed(seed)         # Set random seed

    system_id = args['system_id'] 

    recover_training_flag = args['recover_training_flag']
    
    ### Not tested ###
    #GPU_flag = args['GPU_flag'] 
    #if GPU_flag:
    #    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    #print(tf.config.experimental.list_physical_devices('GPU'))
    
    nb_cpus = args['nb_cpus']

    w_S = args['w_S']
    #########################################################



    # Import configuration file and environment file
    system_map = {
        'single_integrator': ('conf_single_integrator', 'SingleIntegrator', 'SingleIntegrator_CAMS'),
        'double_integrator': ('conf_double_integrator', 'DoubleIntegrator', 'DoubleIntegrator_CAMS'),
        'car':               ('conf_car', 'Car', 'Car_CAMS'),
        'car_park':          ('conf_car_park', 'CarPark', 'CarPark_CAMS'),
        'manipulator':       ('conf_manipulator', 'Manipulator', 'Manipulator_CAMS'),
        'ur5':               ('conf_ur5', 'UR5', 'UR5_CAMS')
    }
    try:
        conf_module, env_class, env_TO_class = system_map[system_id]
        conf = importlib.import_module(conf_module)
        Environment = getattr(importlib.import_module('environment'), env_class)
        Environment_TO = getattr(importlib.import_module('environment_TO'), env_TO_class)
    except KeyError:
        print('System {} not found'.format(system_id))
        sys.exit()

    # Create folders to store the results and the trained NNs
    for path in conf.path_list:
        os.makedirs(path + '/N_try_{}'.format(N_try), exist_ok=True)
    os.makedirs(conf.Config_path, exist_ok=True)

    # Save configuration
    params = [p for p in conf.__dict__ if not p.startswith("__")]
    with open(conf.Config_path + '/config{}.txt'.format(N_try), 'w') as f:
        for p in params:
            f.write('{} = {}\n'.format(p, conf.__dict__[p]))
        f.write('Seed = {}\n'.format(seed))
        f.write('w_S = {}'.format(w_S))

    shutil.copy('{}.py'.format(conf_module), conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try))
    with open(conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try), 'a') as f:
        f.write('\n\n# {}'.format(args))

    # Copy all file with .py extension from /mydir to /mydestdir
    for file in os.listdir("./"):
        if file.endswith(".py"):
            shutil.copy(os.path.join("./", file), os.path.join(conf.Code_path + '/N_try_{}'.format(N_try), file))

    # Create empty txt file in Log_path to store the test info
    open(conf.Log_path + '/info.txt', 'a').close()



    ### Create instances of the used classes ###
    env = Environment(conf)                                                                                 # Create environment instances
    env_TO = Environment_TO
    NN_inst = NN(env, conf, w_S)                                                                            # Create NN instance
    TrOp = TO_Casadi(env, conf, env_TO, w_S)                                                                # Create TO instance
    RLAC = RL_AC(env, NN_inst, conf, N_try)                                                                 # Create RL instance
    buffer = ReplayBuffer(conf) if conf.prioritized_replay_alpha == 0 else PrioritizedReplayBuffer(conf)    # Create an empty (prioritized) replay buffer
    plot_fun = PLOT(N_try, env, NN_inst, conf)                                                              # Create PLOT instance

    # Set initial weights of the NNs, initialize the counter of the updates and setup NN models
    if recover_training_flag:
        recover_training = np.array([conf.NNs_path_rec, conf.N_try_rec, conf.update_step_counter_rec])
        update_step_counter = conf.update_step_counter_rec

        RLAC.setup_model(recover_training)
    else:
        update_step_counter = 0

        RLAC.setup_model()

    # Save initial weights of the NNs
    RLAC.RL_save_weights(update_step_counter)

    # Plot initial rollouts
    plot_fun.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp, RLAC, update_step_counter=update_step_counter,steps=conf.NSTEPS, init=0)

    # Initialize arrays to store the reward history of each episode and the average reward history of last 100 episodes
    ep_arr_idx = 0
    ep_reward_arr = np.zeros(conf.NEPISODES-ep_arr_idx)*np.nan                                                                                     

    def compute_sample(args):
        ''' Create samples solving TO problems starting from given ICS '''
        ep = args[0]
        ICS = args[1]

        # Create initial TO #
        init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH, success_init_flag = RLAC.create_TO_init(ep, ICS)
        if success_init_flag == 0:
            return None
            
        # Solve TO problem #
        TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, dVdx = TrOp.TO_Solve(init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
        if success_flag == 0:
            return None
        
        # Collect experiences 
        state_arr, partial_reward_to_go_arr, total_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr  = RLAC.RL_Solve(TO_controls, TO_states, TO_step_cost)

        if conf.env_RL == 0:
            RL_ee_pos_arr = TO_ee_pos_arr

        return NSTEPS_SH, TO_controls, TO_ee_pos_arr, dVdx, state_arr.tolist(), partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr

    def create_unif_TO_init(n_UICS=1):
        ''' Create n uniformely distributed ICS '''
        # Create ICS TO #
        init_rand_state = env.reset()
        
        return init_rand_state
    


    ### START TRAINING ###
    if conf.profile:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    time_start = time.time()

    for ep in range(conf.NLOOPS): 
        # Generate and store conf.EP_UPDATE random-uniform ICS
        with Pool(nb_cpus) as p: 
            init_rand_state = p.map(create_unif_TO_init, range(conf.EP_UPDATE))

        # Generate samples
        with Pool(nb_cpus) as p: 
            tmp = p.map(compute_sample, zip(ep*np.ones(conf.EP_UPDATE), init_rand_state))
            
        # Remove unsuccessful TO problems and update EP_UPDATE
        tmp = [x for x in tmp if x is not None]
        NSTEPS_SH, TO_controls, ee_pos_arr_TO, dVdx, state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL = zip(*tmp)

        # Update the buffer
        buffer.add(state_arr, partial_reward_to_go_arr, state_next_rollout_arr, dVdx, done_arr, term_arr)

        # Update NNs
        update_step_counter = RLAC.learn_and_update(update_step_counter, buffer, ep)

        # plot Critic value function
        #plot_fun.plot_Critic_Value_function(RLAC.critic_model, update_step_counter, system_id) ###

        # Plot rollouts and state and control trajectories
        if update_step_counter%conf.plot_rollout_interval_diff_loc == 0 or system_id == 'single_integrator' or system_id == 'double_integrator' or system_id == 'car_park' or system_id == 'car' or system_id == 'manipulator':
            print("System: {} - N_try = {}".format(conf.system_id, N_try))
            plot_fun.plot_Critic_Value_function(RLAC.critic_model, update_step_counter, system_id)
            plot_fun.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp, RLAC, update_step_counter=update_step_counter, ep=ep,steps=conf.NSTEPS, init=1)

        # Update arrays to store the reward history and its average
        ep_reward_arr[ep_arr_idx:ep_arr_idx+len(tmp)] = ep_return
        ep_arr_idx += len(tmp)

        for i in range(len(tmp)):
            print("Episode  {}  --->   Return = {}".format(ep*len(tmp) + i, ep_return[i]))

        if update_step_counter > conf.NUPDATES:
            break

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    if conf.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

    # Plot returns
    plot_fun.plot_Return(ep_reward_arr)

    # Save networks at the end of the training
    RLAC.RL_save_weights()

    # Simulate the final policy
    plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim)
