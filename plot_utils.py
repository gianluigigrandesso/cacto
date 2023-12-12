import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d

class PLOT():
    def __init__(self, N_try, env, NN, conf):
        '''    
        :input N_try :                          (Test number)

        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)
            :param fig_ax_lim :                 (float array) Figure axis limit [x_min, x_max, y_min, y_max]
            :param Fig_path :                   (str) Figure path
            :param NSTEPS :                     (int) Max episode length
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param dt :                         (float) Timestep
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
        '''
        self.env = env  
        self.NN = NN     
        self.conf = conf

        self.N_try = N_try

        self.xlim = conf.fig_ax_lim[0].tolist()
        self.ylim = conf.fig_ax_lim[1].tolist()

        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22
        plt.rcParams.update({'font.size': 20})

        return 

    def plot_obstaces(self, a=1):
        if self.conf.system_id == 'car_park':
            obs1 = Rectangle((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Rectangle((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Rectangle((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

            #rec1 = FancyBboxPatch((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec1.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec2 = FancyBboxPatch((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec2.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec3 = FancyBboxPatch((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec3.set_facecolor([30/255, 130/255, 76/255, 1])
        else:
            obs1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

        return [obs1, obs2, obs3]

    def plot_Reward(self, plot_obs=0):
        x = np.arange(-15, 15, 0.1)
        y = np.arange(-10, 10, 0.1)
        theta = np.pi/2
        ICS = np.array([np.array([i,j,0]) for i in x for j in y])
        state = np.array([self.compute_ICS(np.array([i,j,0]), 'car')[0] for i in x for j in y]) # for k in theta]
        state[:,2] = theta
        r = [self.env.reward(self.conf.cost_weights_running, s) for s in state]
        mi = min(r)
        ma = max(r)
        norm = colors.Normalize(vmin=mi,vmax=ma)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        pti = ax.scatter(ICS[:,0], ICS[:,1], norm=norm, c=r, cmap=cm.get_cmap('hot_r'))
        plt.colorbar(pti)

        if plot_obs:
            obs_plot_list = self.plot_obstaces()
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
        
        # Center and check points of 'car_park' system
        #check_points_WF_i = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(self.conf.check_points_BF[0,:]) + ICS[0,:2]
        #ax.scatter(check_points_WF_i[0], check_points_WF_i[1], c='b')
        #for i in range(1,len(self.conf.check_points_BF)):
        #    check_points_WF_i = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(self.conf.check_points_BF[i,:]) + ICS[0,:2]
        #    ax.scatter(check_points_WF_i[0], check_points_WF_i[1], c='r')

        ax.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5, legend='Goal position') 
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Plane')
        #ax.legend()
        ax.grid(True)
        plt.show()
    
    def compute_ICS(self, p_ee, sys_id, theta=None, continue_flag=0):
        if sys_id == 'manipulator':
            radius = math.sqrt((p_ee[0]-self.conf.x_base)**2+(p_ee[1])**2)
            if radius > 30:
                continue_flag = 1
                return None, continue_flag

            phi = math.atan2(p_ee[1]-self.conf.y_base,(p_ee[0]-self.conf.x_base))               # SUM OF THE ANGLES FIXED   
            X3rd_joint = (p_ee[0]-self.conf.x_base) - self.conf.l* math.cos(phi) 
            Y3rd_joint = (p_ee[1]-self.conf.y_base) - self.conf.l* math.sin(phi)

            if abs(X3rd_joint) <= 1e-6 and abs(Y3rd_joint) <= 1e-6:
                continue_flag = 1
                return None, continue_flag

            c2 = (X3rd_joint**2 + Y3rd_joint**2 -2*self.conf.l**2)/(2*self.conf.l**2)

            if p_ee[1] >= 0:
                s2 = math.sqrt(1-c2**2)
            else:
                s2 = -math.sqrt(1-c2**2)

            s1 = ((self.conf.l + self.conf.l*c2)*Y3rd_joint - self.conf.l*s2*X3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)  
            c1 = ((self.conf.l + self.conf.l*c2)*X3rd_joint - self.conf.l*s2*Y3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)
            ICS_q0 = math.atan2(s1,c1)
            ICS_q1 = math.atan2(s2,c2)
            ICS_q2 = phi-ICS_q0-ICS_q1

            ICS = np.array([ICS_q0, ICS_q1, ICS_q2, 0.0, 0.0, 0.0, 0.0])

        elif sys_id == 'car':
            if theta == None:
                theta = 0*np.random.uniform(-math.pi,math.pi)
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'car_park':
            if theta == None:
                #theta = 0*np.random.uniform(-math.pi,math.pi)
                theta = np.pi/2
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'double_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0, 0.0, 0.0])
        
        elif sys_id == 'single_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0])
        
        return ICS, continue_flag

    def plot_policy(self, tau, x, y, steps, n_updates, diff_loc=0):
        ''' Plot policy rollout from a single initial state as well as state and control trajectories '''
        timesteps = self.self.conf.dt*np.arange(steps)
        
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1, legedn='x') 
        ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1, legend='y')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('[m]')  
        ax1.set_title('End-Effector Position') 
        ax1.legend()
        ax1.grid(True) 

        col = ['ro', 'bo', 'go']
        ax2 = fig.add_subplot(2, 2, self.conf.nb_action)
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps, tau[:,i], col[i], linewidth=1, markersize=1,legend='tau{}'.format(i)) 
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Controls')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(x, y, 'ro', linewidth=1, markersize=1) 
        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax3.add_artist(obs_plot_list[i]) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=10) 
        ax3.set_xlim(self.xlim)
        ax3.set_ylim(self.ylim)
        ax3.set_aspect('equal', 'box')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_title('Plane')
        ax3.grid(True)

        fig.tight_layout()

        if diff_loc==0:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}'.format(self.N_try,n_updates))
        else:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}'.format(self.N_try,n_updates))

        plt.clf()
        plt.close(fig)

    def plot_policy_eval(self, p_list, n_updates, diff_loc=0, theta=0):
        ''' Plot only policy rollouts from multiple initial states '''
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1)

        ax = fig.add_subplot(1, 1, 1)
        for idx in range(len(p_list)):
            ax.plot(p_list[idx][:,0], p_list[idx][:,1], marker='o', linewidth=0.3, markersize=1)
            ax.plot(p_list[idx][0,0],p_list[idx][0,1],'ko',markersize=5)
            if self.conf.system_id == 'car_park':
                theta = p_list[idx][-1,2]
                fancybox = FancyBboxPatch((0 - self.conf.L/2, 0 - self.conf.W/2), self.conf.L, self.conf.W, edgecolor='none', alpha=0.5, boxstyle='round,pad=0')
                fancybox.set_transform(Affine2D().rotate_deg(np.rad2deg(theta)).translate(p_list[idx][-1,0], p_list[idx][-1,1]) + ax.transData)
                ax.add_patch(fancybox)

        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax.add_artist(obs_plot_list[i]) 

        ax.plot(self.conf.TARGET_STATE[0],self.conf.TARGET_STATE[1],'b*',markersize=10)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True)
        fig.tight_layout()
        if diff_loc==0:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}'.format(self.N_try,n_updates))
        else:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}'.format(self.N_try,n_updates))

        plt.clf()
        plt.close(fig)

    def rollout(self,update_step_cntr, actor_model, init_states_sim, diff_loc=0):
        ''' Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval() '''
        #tau_all_sim = []
        p_ee_all_sim = []

        returns = {}

        for k in range(len(init_states_sim)):
            rollout_controls = np.zeros((self.conf.NSTEPS,self.conf.nb_action))
            rollout_states = np.zeros((self.conf.NSTEPS+1,self.conf.nb_state))
            rollout_p_ee = np.zeros((self.conf.NSTEPS+1,3))
            rollout_episodic_reward = 0

            rollout_p_ee[0,:] = self.env.get_end_effector_position(init_states_sim[k])
            rollout_states[0,:] = np.copy(init_states_sim[k])
            
            for i in range(self.conf.NSTEPS):
                rollout_controls[i,:] = tf.squeeze(self.NN.eval(actor_model, np.array([rollout_states[i,:]]))).numpy()
                rollout_states[i+1,:], rwrd_sim = self.env.step(self.conf.cost_weights_running, rollout_states[i,:],rollout_controls[i,:])
                rollout_p_ee[i+1,:] = self.env.get_end_effector_position(rollout_states[i+1,:])
                
                rollout_p_ee[i+1,-1] = rollout_states[i+1,2] ### !!! ###

                rollout_episodic_reward += rwrd_sim

            if k==0:
                print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(self.N_try,update_step_cntr,rollout_episodic_reward))
                
            p_ee_all_sim.append(rollout_p_ee)  

            returns[init_states_sim[k][0],init_states_sim[k][1]] = rollout_episodic_reward

        self.plot_policy_eval(p_ee_all_sim,update_step_cntr, diff_loc=diff_loc)

        return returns

    def plot_results(self, tau, ee_pos_TO, ee_pos_RL, steps, to=0):
        ''' Plot results from TO and episode to check consistency '''
        timesteps = self.conf.dt*np.arange(steps+1)
        fig = plt.figure(figsize=(12,8))
        if to:
            plt.suptitle('TO EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)
        else:  
            plt.suptitle('POLICY EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, ee_pos_TO[:,0], 'ro', linewidth=1, markersize=1,legend="x_TO") 
        ax1.plot(timesteps, ee_pos_TO[:,1], 'bo', linewidth=1, markersize=1,legend="y_TO")
        ax1.plot(timesteps, ee_pos_RL[:,0], 'go', linewidth=1, markersize=1,legend="x_RL") 
        ax1.plot(timesteps, ee_pos_RL[:,1], 'ko', linewidth=1, markersize=1,legend="y_RL")
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('[m]')    
        ax1.set_title('End-Effector Position')
        ax1.set_xlim(0, timesteps[-1])
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 3)
        col = ['ro', 'bo', 'go']
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps[:-1], tau[:,i], col[i], linewidth=1, markersize=1,legend='tau{}'.format(i)) 
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Controls')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(ee_pos_TO[:,0], ee_pos_TO[:,1], 'ro', linewidth=1, markersize=2,legend='TO')
        ax3.plot(ee_pos_RL[:,0], ee_pos_RL[:,1], 'bo', linewidth=1, markersize=2,legend='RL')
        ax3.plot([ee_pos_TO[0,0]],[ee_pos_TO[0,1]],'ro',markersize=5)
        ax3.plot([ee_pos_RL[0,0]],[ee_pos_RL[0,1]],'bo',markersize=5)
        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax3.add_artist(obs_plot_list[i]) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
        ax3.set_xlim(self.xlim)
        ax3.set_ylim(self.ylim)
        ax3.set_aspect('equal', 'box')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_title('Plane')
        ax3.legend()
        ax3.grid(True)

        fig.tight_layout()
        #plt.show()

    def plot_Return(self, ep_reward_list):
        ''' Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized) '''
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)   
        ax.set_yscale('log') 
        ax.plot(ep_reward_list**2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("N_try = {}".format(self.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/EpReturn_{}'.format(self.N_try))
        plt.close()

    def plot_Critic_Value_function(self, critic_model, n_update, sys_id, name='V'):
        ''' Plot Value function as learned by the critic '''
        if sys_id == 'manipulator':
            N_discretization_x = 60 + 1  
            N_discretization_y = 60 + 1

            plot_data = np.zeros(N_discretization_y*N_discretization_x)*np.nan
            ee_pos = np.zeros((N_discretization_y*N_discretization_x,3))*np.nan

            for k_x in range(N_discretization_x):
                for k_y in range(N_discretization_y):
                    ICS = self.env.reset()
                    ICS[-1] = 0
                    ee_pos[k_x*(N_discretization_y)+k_y,:] = self.env.get_end_effector_position(ICS)
                    plot_data[k_x*(N_discretization_y)+k_y] = self.NN.eval(critic_model, np.array([ICS]))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()
            plt.scatter(ee_pos[:,0], ee_pos[:,1], c=plot_data, cmap=cm.coolwarm, antialiased=False)
            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

        else:
            N_discretization_x = 30 + 1  
            N_discretization_y = 30 + 1

            plot_data = np.zeros((N_discretization_y,N_discretization_x))*np.nan

            ee_x = np.linspace(-15, 15, N_discretization_x)
            ee_y = np.linspace(-15, 15, N_discretization_y)

            for k_y in range(N_discretization_y):
                for k_x in range(N_discretization_x):
                    p_ee = np.array([ee_x[k_x], ee_y[k_y], 0])
                    ICS, continue_flag = self.compute_ICS(p_ee, sys_id, continue_flag=0)
                    if continue_flag:
                        continue
                    plot_data[k_x,k_y] = self.NN.eval(critic_model, np.array([ICS]))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()
            plt.contourf(ee_x, ee_y, plot_data.T, cmap=cm.coolwarm, antialiased=False)

            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

    def plot_Critic_Value_function_from_sample(self, n_update, NSTEPS_SH, state_arr, reward_arr):
        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        reward_to_go_arr = np.zeros(sum(NSTEPS_SH)+len(NSTEPS_SH)*1)
        idx = 0
        for n in range(len(NSTEPS_SH)):
            for i in range(NSTEPS_SH[n]+1):
                # Compute the partial cost to go
                reward_to_go_arr[idx] = sum(reward_arr[n][i:])
                idx += 1

        state_arr = np.concatenate(state_arr, axis=0)
        ee_pos_arr = np.zeros((len(state_arr),3))
        for i in range(state_arr.shape[0]):
            ee_pos_arr[i,:] = self.env.get_end_effector_position(state_arr[i])
        

        mi = min(reward_to_go_arr)
        ma = max(reward_to_go_arr)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()#projection='3d')
        norm = colors.Normalize(vmin=mi,vmax=ma)

        obs_plot_list = self.plot_obstaces(a=0.5)
        
        ax.scatter(ee_pos_arr[:,0],ee_pos_arr[:,1], c=reward_to_go_arr, norm=norm, cmap=cm.coolwarm, marker='x')
        
        for i in range(len(obs_plot_list)):
            ax.add_patch(obs_plot_list[i])

        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.coolwarm))
        plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        plt.savefig('{}/N_try_{}/V_sample_{}'.format(self.conf.Fig_path,self.N_try,int(n_update)))
        plt.close()

    def plot_ICS(self,state_arr):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        for j in range(len(state_arr)):
            ax.scatter(state_arr[j][0,0],state_arr[j][0,1])
            obs_plot_list = plot_fun.plot_obstaces()
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
        ax.set_xlim(self.fig_ax_lim[0].tolist())
        ax.set_ylim(self.fig_ax_lim[1].tolist())
        ax.set_aspect('equal', 'box')
        plt.savefig('{}/N_try_{}/ICS_{}_S{}'.format(conf.Fig_path,N_try,update_step_counter,int(w_S)))
        plt.close(fig)

    def plot_rollout_and_traj_from_ICS(self, init_state, n_update, actor_model, TrOp, tag, steps=200):
        ''' Plot results from TO and episode to check consistency '''
        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        
        for j in range(len(init_state)):

            ee_pos_TO = np.zeros((steps,3))
            ee_pos_RL = np.zeros((steps,3))

            RL_states = np.zeros((steps,self.conf.nb_state))
            RL_action = np.zeros((steps-1,self.conf.nb_action))
            RL_states[0,:] = init_state[j,:]
            ee_pos_RL[0,:] = self.env.get_end_effector_position(RL_states[0,:])

            for i in range(steps-1):
                RL_action[i,:] = self.NN.eval(actor_model, np.array([RL_states[i,:]]))
                RL_states[i+1,:] = self.env.simulate(RL_states[i,:], RL_action[i,:])
                ee_pos_RL[i+1,:] = self.env.get_end_effector_position(RL_states[i+1,:])
            
            TO_states, _ = TrOp.TO_System_Solve3(init_state[j,:], RL_states.T, RL_action.T, steps-1)

            try:
                for i in range(steps):
                    ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[i,:])
            except:
                ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[0,:])
                
            ax.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax.scatter(ee_pos_TO[0,0],ee_pos_TO[0,1],color=colors[j])
            ax.scatter(ee_pos_RL[0,0],ee_pos_RL[0,1],color=colors[j])
            ax.plot(ee_pos_TO[1:,0],ee_pos_TO[1:,1],color=colors[j])
            ax.plot(ee_pos_RL[1:,0],ee_pos_RL[1:,1],'--',color=colors[j])
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax.add_patch(obs_plot_list[i])

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Plane')
        #ax.legend()
        ax.grid(True)

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}'.format(self.conf.Fig_path,self.N_try,int(n_update), tag))

    def plot_ICS(self, input_arr, cs=0):
        if cs == 1:
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()
            for j in range(len(input_arr)):
                p_arr[j,:] = input_arr[j,:]
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/ICS'.format(self.conf.Fig_path,self.N_try))
            plt.close(fig)
        else:    
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()

            for j in range(len(input_arr)):
                p_arr[j,:] = self.env.get_end_effector_position(input_arr[j])
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/ICS'.format(self.conf.Fig_path,self.N_try))
            plt.close(fig)

    def plot_traj_from_ICS(self, init_state, TrOp, RLAC, update_step_counter=0,ep=0,steps=200, init=0,continue_flag=1):
        ''' Plot results from TO and episode to check consistency '''
        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        for j in range(len(init_state)):

            ee_pos_TO = np.zeros((steps,3))
            ee_pos_RL = np.zeros((steps,3))
            
            if init == 0:
                # zeros
                _, init_TO_states, init_TO_controls, _, success_init_flag = RLAC.create_TO_init(0, init_state[j,:])
            elif init == 1:
                # NN
                _, init_TO_states, init_TO_controls, _, success_init_flag = RLAC.create_TO_init(1, init_state[j,:])

            if success_init_flag:
                _, _, TO_states, _, _, _  = TrOp.TO_System_Solve(init_state[j,:], init_TO_states, init_TO_controls, steps-1)
            else:
                continue

            try:
                for i in range(steps):
                    ee_pos_RL[i,:] = self.env.get_end_effector_position(init_TO_states[i,:])
                    ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[i,:])
            except:
                ee_pos_RL[i,:] = self.env.get_end_effector_position(init_TO_states[0,:])
                ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[0,:])

            ax1.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax1.scatter(ee_pos_RL[0,0],ee_pos_RL[0,1],color=colors[j])
            ax1.plot(ee_pos_RL[1:,0],ee_pos_RL[1:,1],'--',color=colors[j])
                
            ax2.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax2.scatter(ee_pos_TO[0,0],ee_pos_TO[0,1],color=colors[j])
            ax2.plot(ee_pos_TO[1:,0],ee_pos_TO[1:,1],color=colors[j])
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax1.add_patch(obs_plot_list[i])

        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax2.add_patch(obs_plot_list[i])

        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Warmstart traj.')

        ax2.set_xlim(self.xlim)
        ax2.set_ylim(self.ylim)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X [m]')
        #ax2.set_ylabel('Y [m]')
        ax2.set_title('TO traj.')
        #ax.legend()
        ax1.grid(True)
        ax2.grid(True)

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}'.format(self.conf.Fig_path,self.N_try,init,update_step_counter))


        

if __name__ == '__main__':
    import os
    import sys
    import time
    import random
    import importlib
    import numpy as np
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.art3d as art3d

    from RL import RL_AC 
    from plot_utils import PLOT
    from NeuralNetwork import NN

    ###           Input           ###
    N_try = 0

    seed = 0
    tf.random.set_seed(seed)  # Set tensorflow seed
    random.seed(seed)         # Set random seed

    system_id = 'car_park'

    TO_method = 'casadi'

    recover_training_flag = 0
    
    CPU_flag = 0
    if CPU_flag:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    tf.config.experimental.list_physical_devices('GPU')
    
    nb_cpus = 1

    w_S = 0
    #################################

    # Import configuration file and environment file
    system_map = {
        'single_integrator': ('conf_single_integrator', 'SingleIntegrator'),
        'double_integrator': ('conf_double_integrator', 'DoubleIntegrator'),
        'car':               ('conf_car', 'Car'),
        'car_park':          ('conf_car_park', 'CarPark'),
        'manipulator':       ('conf_manipulator', 'Manipulator'),
        'ur5':               ('conf_ur5', 'UR5')
    }

    try:
        conf_module, env_class = system_map[system_id]
        conf = importlib.import_module(conf_module)
        Environment = getattr(importlib.import_module('environment'), env_class)
    except KeyError:
        print('System {} not found'.format(system_id))
        sys.exit()

        

    # Create folders to store the results and the trained NNs and save configuration
    for path in conf.path_list:
        os.makedirs(path + '/N_try_{}'.format(N_try), exist_ok=True)
    os.makedirs(conf.Config_path, exist_ok=True)

    params = [p for p in conf.__dict__ if not p.startswith("__")]
    with open(conf.Config_path + '/config{}.txt'.format(N_try), 'w') as f:
        for p in params:
            f.write('{} = {}\n'.format(p, conf.__dict__[p]))
        f.write('Seed = {}\n'.format(seed))
        f.write('w_S = {}'.format(w_S))



    # Create environment instances
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

    # Create PLOT instance
    plot_fun = PLOT(N_try, env, NN_inst, conf)

    plot_fun.plot_Reward()