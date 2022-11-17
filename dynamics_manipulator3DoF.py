import os
import math
import gepetto.corbaserver
import time
import subprocess
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper as PinocchioRobotWrapper
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import random

class Contact:
    ''' A contact between a contact-point and a contact-surface
    '''
    def __init__(self, contact_point, contact_surface):
        self.cp = contact_point
        self.cs = contact_surface
        self.reset_contact_position()

    def reset_contact_position(self):
        # Initialize anchor point p0, that is the initial (0-load) position of the spring
        self.p0 = self.cp.get_position()
        self.in_contact = True

    def compute_force(self):
        self.f, self.p0 = self.cs.compute_force(self.cp, self.p0)
        return self.f
        
    def get_jacobian(self):
        return self.cp.get_jacobian()

class RobotWrapper(PinocchioRobotWrapper):
    
    @staticmethod
    def BuildFromURDF(filename, package_dirs=None, root_joint=None, verbose=False, meshLoader=None):
        robot = RobotWrapper()
        robot.initFromURDF(filename, package_dirs, root_joint, verbose, meshLoader)
        return robot
    
    @property
    def na(self):
        if(self.model.joints[0].nq==7):
            return self.model.nv-6
        return self.model.nv

    def mass(self, q, update=True):
        if(update):
            return pin.crba(self.model, self.data, q)
        return self.data.M

    def nle(self, q, v, update=True):
        if(update):
            return pin.nonLinearEffects(self.model, self.data, q, v)
        return self.data.nle
        
    def com(self, q=None, v=None, a=None, update=True):
        if(update==False or q is None):
            return PinocchioRobotWrapper.com(self, q)
        if a is None:
            if v is None:
                return PinocchioRobotWrapper.com(self, q)
            return PinocchioRobotWrapper.com(self, q, v)
        return PinocchioRobotWrapper.com(self, q, v,a)
        
    def Jcom(self, q, update=True):
        if(update):
            return pin.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom
        
    def momentumJacobian(self, q, v, update=True):
        if(update):
            pin.ccrba(self.model, self.data, q, v)
        return self.data.Ag

    def computeAllTerms(self, q, v):
        ''' pin.computeAllTerms is equivalent to calling:
            pinocchio::forwardKinematics
            pinocchio::crba
            pinocchio::nonLinearEffects
            pinocchio::computeJointJacobians
            pinocchio::centerOfMass
            pinocchio::jacobianCenterOfMass
            pinocchio::kineticEnergy
            pinocchio::potentialEnergy
            This is too much for our needs, so we call only the functions
            we need, including those for the frame kinematics
        '''
        #pin.computeAllTerms(self.model, self.data, q, v);
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)
        pin.nonLinearEffects(self.model, self.data, q, v)
       
    def forwardKinematics(self, q, v=None, a=None):
        if v is not None:
            if a is not None:
                pin.forwardKinematics(self.model, self.data, q, v, a)
            else:
                pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)

    def inverseKinematics(self, ee_coords):
        ''' Inverse kinematics algorithm to compute a joint configuration given the EE coordinates'''
        
        oMdes = pin.SE3(np.eye(3), np.array([ee_coords[0], ee_coords[1], 0.0]))
        for j in range(100):
            q = np.array([random.uniform(-math.pi,math.pi),random.uniform(-math.pi,math.pi),random.uniform(-math.pi,math.pi)])
            # q = pin.neutral(self.model)

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            eps    = 1e-4
            IT_MAX = 1000
            DT     = 1e-1
            damp   = 1e-12
            frame_id = self.model.getFrameId('fixed_ee')    
            i=0
            while True:
                pin.forwardKinematics(self.model,self.data,q)
                pin.updateFramePlacements(self.model, self.data)
                dMi = oMdes.actInv(self.data.oMf[frame_id])
                err = pin.log(dMi).vector
                if norm(err) < eps:
                    success = True
                    break
                if i >= IT_MAX:
                    success = False
                    break
                J = pin.computeFrameJacobian(self.model,self.data,q,frame_id)
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model,q,v*DT)
                if not i % 10:
                    print('%d: error = %s' % (i, err.T))
                i += 1
            
            if success:
                print("Convergence achieved!")
                break
            else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision. Retry ({}) with another initial configuration".format(j))
                
        return q, success

    def frameJacobian(self, q, index, update=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        ''' Call computeFrameJacobian if update is true. If not, user should call computeFrameJacobian first.
            Then call getFrameJacobian and return the Jacobian matrix.
            ref_frame can be: ReferenceFrame.LOCAL, ReferenceFrame.WORLD, ReferenceFrame.LOCAL_WORLD_ALIGNED
        '''
        if(update): 
            pin.computeFrameJacobian(self.model, self.data, q, index)
        return pin.getFrameJacobian(self.model, self.data, index, ref_frame)
        
    def frameVelocity(self, q, v, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v)
        v_local = pin.getFrameVelocity(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return v_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            v_world = H.act(v_local)
            return v_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        v = Hr.act(v_local)
        return v
            
    def frameAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
        
    def frameClassicAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        v = pin.getFrameVelocity(self.model, self.data, index)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        a_local.linear += np.cross(v.angular, v.linear, axis=0)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
      
    def deactivateCollisionPairs(self, collision_pair_indexes):
        for i in collision_pair_indexes:
            self.collision_data.deactivateCollisionPair(i)
            
    def addAllCollisionPairs(self):
        self.collision_model.addAllCollisionPairs()
        self.collision_data = pin.GeometryData(self.collision_model)
        
    def isInCollision(self, q, stop_at_first_collision=True):
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, np.asmatrix(q).reshape((self.model.nq,1)), stop_at_first_collision)

    def findFirstCollisionPair(self, consider_only_active_collision_pairs=True):
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    return (i, self.collision_model.collisionPairs[i])
        return None
        
    def findAllCollisionPairs(self, consider_only_active_collision_pairs=True):
        res = []
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    res += [(i, self.collision_model.collisionPairs[i])]
        return res

class RobotSimulator:

    def __init__(self, robot, q0_init, v0_init, simulation_type, tau_coulomb_max, use_viewer=False, DISPLAY_T=None, CAMERA_TRANSFORM=None, show_floor=False):
        self.robot = robot
        self.model = self.robot.model
        self.data = self.model.createData()
        self.t = 0.0                            # time
        self.nv = nv = self.model.nv            # Dimension of joint velocities vector
        self.na = na = robot.na                 # number of actuated joints
        self.S = np.hstack((np.zeros((na, nv-na)), np.eye(na, na))) # Matrix S used as filter of vector of inputs U

        self.contacts = []
        self.candidate_contact_points = []      # candidate contact points
        self.contact_surfaces = []
        
        self.DISPLAY_T = DISPLAY_T              # refresh period for viewer
        self.use_viewer = use_viewer
        self.tau_coulomb_max = tau_coulomb_max
        self.display_counter = self.DISPLAY_T
        self.init(q0_init, v0_init, True)
        
        self.tau_c = np.zeros(na)   # Coulomb friction torque
        self.simulation_type = simulation_type
        self.set_coulomb_friction(tau_coulomb_max)

        # for gepetto viewer
        self.gui = None
        if(use_viewer):
            try:
                prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(prompt[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
            except:
                pass
            gepetto.corbaserver.Client()
            self.robot.initViewer(loadModel=True)
            self.gui = self.robot.viewer.gui
            if(show_floor):
                self.robot.viewer.gui.createSceneWithFloor('world')
                self.gui.setLightingMode('world/floor', 'OFF')
            self.robot.displayCollisions(False)
            self.robot.displayVisuals(True)
            self.robot.display(self.q)            
            try:  
                self.gui.setCameraTransform("python-pinocchio", CAMERA_TRANSFORM)
            except:
                self.gui.setCameraTransform(0, CAMERA_TRANSFORM)            

    # Re-initialize the simulator
    def init(self, q0=None, v0=None, reset_contact_positions=False):
        self.first_iter = True

        if q0 is not None:
            self.q = q0.copy()
            
        if(v0 is None):
            self.v = np.zeros(self.robot.nv)
        else:
            self.v = v0.copy()
        self.dv = np.zeros(self.robot.nv)
        self.resize_contact_data(reset_contact_positions)
          
    def resize_contact_data(self, reset_contact_positions=False):
        self.nc = len(self.contacts)                    # number of contacts
        self.nk = 3*self.nc                             # size of contact force vector
        self.f = np.zeros(self.nk)                      # contact forces
        self.Jc = np.zeros((self.nk, self.model.nv))    # contact Jacobian

        # reset contact position
        if(reset_contact_positions):
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            for c in self.contacts:
                c.reset_contact_position()

        self.compute_forces(compute_data=True)
        
    def set_coulomb_friction(self, tau_max):
        self.tau_coulomb_max = 1e-2*tau_max*self.model.effortLimit        
        self.simulate_coulomb_friction = (norm(self.tau_coulomb_max)!=0.0)
        
    def collision_detection(self):
        for s in self.contact_surfaces:                 # for each contact surface
            for cp in self.candidate_contact_points:    # for each candidate contact point
                p = cp.get_position()
                if(s.check_collision(p)):               # check whether the point is colliding with the surface
                    if(not cp.active):                  # if the contact was not already active
                        print("Collision detected between point", cp.frame_name, " at ", p)
                        cp.active = True
                        cp.contact = Contact(cp, s)
                        self.contacts += [cp.contact]
                        self.resize_contact_data()
                else:
                    if(cp.active):
                        print("Contact lost between point", cp.frame_name, " at ", p)
                        cp.active = False
                        self.contacts.remove(cp.contact)
                        self.resize_contact_data()

    def compute_forces(self, compute_data=True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data:
            pin.forwardKinematics(self.model, self.data, self.q, self.v)
            # Computes the placements of all the operational frames according to the current joint placement stored in data
            pin.updateFramePlacements(self.model, self.data)
            self.collision_detection()
            
        i = 0
        for c in self.contacts:
            self.f[i:i+3] = c.compute_force()
            self.Jc[i:i+3, :] = c.get_jacobian()
            i += 3
        return self.f

    def step(self, u, dt=None):
        if dt is None:
            dt = self.dt

        # (Forces are directly in the world frame, and aba wants them in the end effector frame)
        pin.computeAllTerms(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        M = self.data.M
        h = self.data.nle
        self.collision_detection()
        self.compute_forces(False)

        if(self.simulate_coulomb_friction and self.simulation_type=='timestepping'):
            # minimize kinetic energy using time stepping
            from quadprog import solve_qp
            '''
            Solve a strictly convex quadratic program
            
            Minimize     1/2 x^T G x - a^T x
            Subject to   C.T x >= b
            
            Input Parameters
            ----------
            G : array, shape=(n, n)
            a : array, shape=(n,)
            C : array, shape=(n, m) matrix defining the constraints
            b : array, shape=(m), default=None, vector defining the constraints
            meq : int, default=0
                the first meq constraints are treated as equality constraints,
                all further as inequality constraints
            '''
            # M (v' - v) = dt*S^T*(tau - tau_c) - dt*h + dt*J^T*f
            # M v' = M*v + dt*(S^T*tau - h + J^T*f) - dt*S^T*tau_c
            # M v' = b + B*tau_c
            # v' = Minv*(b + B*tau_c)
            b = M.dot(self.v) + dt*(self.S.T.dot(u) - h + self.Jc.T.dot(self.f))
            B = - dt*self.S.T
            # Minimize kinetic energy:
            # min v'.T * M * v'
            # min  (b+B*tau_c​).T*Minv*(b+B*tau_c​) 
            # min tau_c.T * B.T*Minv*B* tau_C + 2*b.T*Minv*B*tau_c
            Minv = np.linalg.inv(M)
            G = B.T.dot(Minv.dot(B))
            a = -b.T.dot(Minv.dot(B))
            C = np.vstack((np.eye(self.na), -np.eye(self.na)))
            c = np.concatenate((-self.tau_coulomb_max, -self.tau_coulomb_max))
            solution = solve_qp(G, a, C.T, c, 0)
            self.tau_c = solution[0]
            self.v = Minv.dot(b + B.dot(self.tau_c))
            self.q = pin.integrate(self.model, self.q, self.v*dt)
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h + self.Jc.T.dot(self.f))
        elif(self.simulation_type=='euler' or self.simulate_coulomb_friction==False):
            self.tau_c = self.tau_coulomb_max*np.sign(self.v[-self.na:])
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h + self.Jc.T.dot(self.f))
            # v_mean = np.copy(self.v) + 0.5*dt*self.dv
            self.q = pin.integrate(self.model, self.q, self.v*dt)
            # self.q += dt*v_mean        
            self.v += self.dv*dt
        else:
            print("[ERROR] Unknown simulation type:", self.simulation_type)

        self.t += dt
        return self.q, self.v, self.dv

    def reset(self):
        self.first_iter = True

    def simulate(self, u, dt=0.05, ndt=1):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
        tau_c_avg = 0*self.tau_c
        for i in range(ndt):
            self.q, self.v, self.dv = self.step(u, dt/ndt)
            tau_c_avg += self.tau_c
        self.tau_c = tau_c_avg / ndt

        if (self.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0.0:
                self.robot.display(self.q)
                self.display_counter = self.DISPLAY_T

        return self.q, self.v, self.f
        
    def display(self, q):
        if(self.use_viewer):
            self.robot.display(q)

def load_urdf(URDF_PATH):
    robot = RobotWrapper.BuildFromURDF(URDF_PATH)
    return robot

def create_empty_figure(nRows=1, nCols=1,sharex=True):
    f, ax = plt.subplots(nRows,nCols,sharex=sharex)
    return (f, ax)


if __name__=='__main__':

    N = 200                                 # Number of time steps
    dt = 0.05                               # Controller time step
    T_SIMULATION = N*dt                     # Number of seconds simulated
    ndt = 1
    q_init = np.array([math.pi/2,0.,0.])    # Initial joint position
    v_init = np.array([ 0.,0.,0.]).T        # Initial joint velocity

    simulate_coulomb_friction = 0           # To simulate friction
    simulation_type = 'euler'               # Either 'timestepping' or 'euler'
    tau_coulomb_max = 0*np.ones(3)          # Expressed as percentage of torque max

    use_viewer = True
    simulate_real_time = True
    show_floor = False
    PRINT_T = 1                             # Print every PRINT_N time steps
    DISPLAY_T = 0.02                        # Update robot configuration in viewer every DISPLAY_N time steps
    CAMERA_TRANSFORM = [3.3286049365997314, -11.498767852783203, 121.38613891601562, 0.051378559321165085, 0.023286784067749977, -0.001170438714325428, 0.9984070062637329]

    PLOT_JOINT_POS = 1
    PLOT_JOINT_VEL = 1
    PLOT_JOINT_ACC = 1
    PLOT_TORQUES = 1

    URDF_PATH = "urdf/planar_manipulator_3dof.urdf"

    r = load_urdf(URDF_PATH)
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max, use_viewer, DISPLAY_T, CAMERA_TRANSFORM, show_floor)

    frame_id = robot.model.getFrameId('fixed_ee')   # End-effector frame

    # Example of TO solution from [np.array([pi/2,0.,0.,0.,0.,0.])], target=[-20,0]
    tau0 =  [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 3.913573721018818, -49.99999997812837, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -49.99999998263544, -49.99999995861845, -49.99999993221877, -49.99999990306833, -49.99999987071753, -49.99999983461132, -49.9999997940566, -49.99999974817655, -49.999999695845204, -49.999999635591124, -49.999999565453635, -49.99999948277056, -49.9999993838735, -49.99999926362621, -49.99999911450014, -49.99999892406386, -49.99999866788871, -49.999998291144024, -49.99999766106636, -49.99999641971001, -49.99999334107301, -49.99998117668781, -46.3305280659364, -24.952363144765123, -19.338494352862362, -17.665869556802168, -15.969375750675875, -13.875635089856688, -11.802969516419605, -10.018418774230609, -8.566108149374008, -7.391570403941093, -6.42809739367328, -5.62500889060553, -4.94790811736938, -4.3727372800157545, -3.8813594725379144, -3.459361037489522, -3.0950636583315094, -2.7789782267165024, -2.5033915044260913, -2.262015684970941, -2.049698117228155, -1.8621904342485147, -1.6959687118390823, -1.5480935908624742, -1.4161006130642122, -1.2979134200206421, -1.1917745049867159, -1.0961896285185433, -1.0098829686449564, -0.9317607540079113, -0.8608816318066894, -0.7964324077286956, -0.7377080914713674, -0.6840954101556022, -0.6350591283370008, -0.59013065006311, -0.5488984844539794, -0.5110002398113251, -0.47611587644719633, -0.443962000211009, -0.4142870197805363, -0.38686702350141183, -0.3615022576843117, -0.33801410955749306, -0.31624251481435955, -0.2960437238304552, -0.2772883714424923, -0.25985980487578647, -0.24365263123064845, -0.22857145298785428, -0.21452976412923114, -0.20144898449681217, -0.18925761312116307, -0.17789048415522468, -0.1672881117027469, -0.1573961116191082, -0.14816469026299792, -0.1395481916748371, -0.13150469555377498, -0.12399565976918234, -0.11698560197296447, -0.11044181542538248, -0.10433411498268876, -0.0986346096734115, -0.09331749870240803, -0.08835888827625434, -0.08373662679008599, -0.07943015630294098, -0.0754203785771209, -0.07168953402935123, -0.06822109216157565, -0.06499965235029037, -0.06201085377818968, -0.05924129365424839, -0.05667845278067173, -0.05431062782792054, -0.05212686954785943, -0.05011692632728752, -0.04827119268177411, -0.046580662063378186, -0.045036883658417244, -0.043631922768426736, -0.042358324419103935, -0.041209079956523405, -0.040177596288934546, -0.03925766757960086, -0.03844344915052737, -0.03772943342401835, -0.037110427659416016, -0.0365815334480049, -0.03613812768636835, -0.0357758449797231, -0.03549056133594179, -0.03527837907343018, -0.03513561283161326, -0.03505877652434707, -0.03504457090814823, -0.035089871084497695, -0.035191714362526536, -0.035347293877997926, -0.03555397404494562, -0.035809340997228704, -0.0361112098283853, -0.036457214815460895, -0.03684329262668612, -0.03726157437932053, -0.03770484714126743, -0.03819829113291652, -0.03887398161427306, -0.03998048529109933, -0.04134077898704407, -0.04040053594758688, -0.02937217479892824, 9.310123181419991e-22, 3.2867662109711427e-21] 
    tau1 =  [-50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -49.999999925160694, -49.99999623355654, 49.99999661317066, 49.99999933991629, 49.99999958845886, 49.99999953495631, 49.9999991795935, 49.99999757171713, -13.88363883189849, -49.999997992976596, -49.99999919548091, -49.999999489843475, -49.99999954363985, -49.9999993577583, -49.99999849772317, -29.756444147424194, 49.99999851297247, 49.99999962440779, 49.9999999322578, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 49.999999540051434, 49.999945429392575, 13.523097736728628, 49.99999982418196, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 49.99999984633092, 49.99999847815977, 45.367144514865466, -33.83630427701238, -13.867233954583172, -0.5150175332480839, 1.6956169602865967, 0.9012440381465341, 0.3187378661553381, 0.26707626201201495, 0.5171759442383039, 0.9285422723436255, 1.4510231388471262, 2.06302507674324, 2.7476945320290667, 3.4893591250457145, 4.273988285544616, 5.089016044958304, 5.921661917254661, 6.756124677893429, 7.574948431143914, 8.379001318796224, 9.236308477361941, 10.315414824693468, 11.75923166010499, 13.268471377299782, 13.62397567301964, 10.907637822540988, 3.969572656941659, -5.830225659311305, -14.779022841260165, -19.47803573384028, -19.589654048970615, -17.768356513401585, -16.36754183864119, -15.36199581612403, -14.35125645595521, -13.244510677137141, -12.135092802473562, -11.11344736797613, -10.211734347258979, -9.423094529313513, -8.72906392583842, -8.113100899129789, -7.5631701490718495, -7.070407986273108, -6.627657375747211, -6.228745775955465, -5.868258643336417, -5.5414765828689125, -5.244321457631866, -4.973282798621214, -4.725337382788586, -4.497875009074871, -4.288635258573854, -4.095654894536202, -3.9172240426050853, -3.751849416948938, -3.5982233037430147, -3.455197342753047, -3.321760340634281, -3.19701947317666, -3.0801843310492707, -2.970553349808765, -2.867502240312227, -2.770474100184701, -2.6789709408909914, -2.5925464096857915, -2.51079952199875, -2.4333692507602214, -2.359929843838086, -2.2901867617590623, -2.223873145112396, -2.1607467353941847, -2.1005871847778277, -2.0431937006102228, -1.9883829781302658, -1.9359873826438572, -1.8858533472115067, -1.8378399580151306, -1.791817702152308, -1.7476673580051276, -1.705279009121051, -1.6645511668765645, -1.625389988380853, -1.5877085778626319, -1.5514263618540947, -1.5164685291262032, -1.4827655280009002, -1.4502526146865036, -1.4188694463462252, -1.3885597141467942, -1.3592708120698516, -1.3309535372229344, -1.303561818469381, -1.2770524703289947, -1.2513849693768573, -1.2265212510808312, -1.202425524762064, -1.1790641047486285, -1.1564052564097662, -1.1344190554708453, -1.1130772590507383, -1.092353187839831, -1.0722216177572568, -1.052658680675167, -1.033641772964263, -1.0151494715756355, -0.9971614566465445, -0.9796584399147069, -0.9626220990198487, -0.946035016537702, -0.9298806236668248, -0.9141431481429039, -0.8988075658099768, -0.8838595559304274, -0.8692854595130881, -0.8550722406754522, -0.8412074506631448, -0.8276791944692483, -0.814476099441573, -0.8015872864325896, -0.7890023425591132, -0.7767112958181723, -0.7647045914228321, -0.7529730698099669, -0.7415079462538057, -0.730300791389307, -0.7193435108300817, -0.7086283186315634, -0.6981477071505003, -0.6878944562286702, -0.6778618247056026, -0.6680440769047711, -0.6584367080561528, -0.649032784005356, -0.6398077540538428, -0.630695334893884, -0.6216287801285251, -0.6128950581280358, -0.6060640106567209, -0.6043160857918533, -0.6056359031766879, -0.5752845647421657, -0.40450819960279844, -5.263219031959085e-22, -1.5380345359532698e-21] 
    tau2 =  [-50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, 15.176472068098633, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 49.99999934388885, -29.77803281309609, -16.221752389165278, 4.122727684604918, 4.166393767837994, 3.514732370695024, 3.5692856129789194, 2.4256753138456904, -1.452743994278073, -1.2080631473419272, -0.16824452899613926, 0.029834958070621483, -0.09106383508912538, -0.21308198787284016, -0.29430046226428763, -0.35447656421814966, -0.40657909246598867, -0.45473497787083195, -0.5002374257267803, -0.5439659948642026, -0.5867233286245945, -0.6291705892311026, -0.6718331473214855, -0.7151390073252673, -0.7592564187013162, -0.8036391351722044, -0.8475769438021128, -0.8952290113173695, -0.9649183468551131, -1.0844424965204236, -1.2399487595565644, -1.2931714157860015, -0.995371696188642, -0.22354939862974435, 0.7437711833682358, 1.3358156125712366, 1.1994824770073702, 0.7863938473922568, 1.3479870385766368, 1.4423112083688652, 1.2691014140109311, 1.0840586470440485, 0.9515668250469531, 0.8585881498965094, 0.785163907513267, 0.7211673137280206, 0.6636031227311561, 0.6119758492465908, 0.566025511310649, 0.5252350824888673, 0.4889536610814826, 0.4565551896804238, 0.42750295577466474, 0.40135140687931065, 0.3777287206766371, 0.3563203706244419, 0.336857711997533, 0.31911016327731473, 0.30287928134679887, 0.28799387108429686, 0.2743058332244163, 0.26168665101999106, 0.2500244422575814, 0.23922149608186133, 0.22919221341436574, 0.21986138001835878, 0.21116271364150332, 0.20303763764195223, 0.19543424288068909, 0.18830640613489275, 0.181613039559615, 0.17531744973111885, 0.1693867885113119, 0.16379158166933863, 0.15850532265275655, 0.1535041219366205, 0.14876640321139548, 0.1442726397412392, 0.14000512485046981, 0.13594777169912195, 0.1320859379448401, 0.128406272477954, 0.12489658021600719, 0.1215457035178347, 0.11834341711682667, 0.11528033564763057, 0.11234783120603883, 0.10953796063043401, 0.10684340052928179, 0.10425738941269863, 0.10177367615415638, 0.09938647369364768, 0.09709041781637545, 0.09488052996924161, 0.09275218385042916, 0.09070107543134344, 0.08872319590873598, 0.08681480716783921, 0.08497241977806003, 0.08319277288486598, 0.08147281610339675, 0.07980969304210336, 0.07820072634929624, 0.07664340395951144, 0.07513536677545843, 0.07367439736441252, 0.07225840936640654, 0.07088543832539371, 0.06955363273033474, 0.06826124625714236, 0.06700663034399097, 0.06578822753486432, 0.0646045653322765, 0.06345425040158229, 0.06233596348969316, 0.061248454517693826, 0.06019053807870064, 0.059161089434241546, 0.0581590405595334, 0.057183376768826046, 0.05623313332378255, 0.05530739245838624, 0.05440528053964406, 0.05352596548789028, 0.05266865427903696, 0.05183259077469347, 0.05101705355436031, 0.05022135390940832, 0.049444834134151606, 0.04868686570280136, 0.04794684774067304, 0.04722420541324774, 0.04651838864563251, 0.04582887084885529, 0.04515514861748155, 0.044496741667771206, 0.0438531899725731, 0.04322403595870467, 0.04260877802808388, 0.04200684603254139, 0.041417895698825716, 0.040843079005469345, 0.04028715618387913, 0.039755323695318014, 0.0392231914332549, 0.03855375230108693, 0.03747277495639135, 0.036335197108619516, 0.03874870351250417, 0.04599993272961439, -1.1268571878427161e-21, -1.7791824201931178e-21]

    tau = [tau0,tau1,tau2]

    t = 0.0
    q = np.zeros((3,N+1))
    v = np.zeros((3,N+1))
    dv = np.zeros((3,N+1))
    q[:,0] = q_init
    v[:,0] = v_init

    # Simulation
    for i in range(len(tau0)):
        time_start = time.time()  
        
        # Send joint torques to simulator
        simu.simulate([tau0[i],tau1[i],tau2[i]], dt, ndt)
        q[:,i], v[:,i], dv[:,i] = simu.q, simu.v, simu.dv    
        t += dt
            
        time_spent = time.time() - time_start
        if(simulate_real_time and time_spent < dt): 
            time.sleep(dt-time_spent)

    # PLOT STUFF
    time = np.arange(0.0, N*dt+dt, dt)

    if PLOT_JOINT_POS:    
        (f, ax) = create_empty_figure(int(robot.nv/2),3)
        ax = ax.reshape(robot.nv)
        for i in range(robot.nv):
            ax[i].plot(time, q[i,:], label='q')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
            
    if PLOT_JOINT_VEL:    
        (f, ax) = create_empty_figure(int(robot.nv/2),3)
        ax = ax.reshape(robot.nv)
        for i in range(robot.nv):
            ax[i].plot(time, v[i,:], label='v')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'v_'+str(i)+' [rad/s]')
            
    if PLOT_JOINT_ACC:    
        (f, ax) = create_empty_figure(int(robot.nv/2),3)
        ax = ax.reshape(robot.nv)
        for i in range(robot.nv):
            ax[i].plot(time, dv[i,:], label=r'$\dot{v}$')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'$\dot{v}_'+str(i)+'$ [rad/s^2]')
    
    if PLOT_TORQUES:    
        (f, ax) = create_empty_figure(int(robot.nv/2),3)
        ax = ax.reshape(robot.nv)
        for i in range(robot.nv):
            ax[i].plot(time, tau[i], label=r'$\tau_$ '+str(i))
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'$tau_'+str(i)+'$ [Nm]')
            
    plt.show()
