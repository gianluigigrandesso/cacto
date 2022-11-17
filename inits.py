''' Init functions to warm-start TO (Pyomo needs python functions to warm-start its variables)
'''

# Warmstarting with CACTO rollout

def init_tau0(m,k,init_TO_controls):
    return init_TO_controls[0][k]

def init_tau1(m,k,init_TO_controls):
    return init_TO_controls[1][k]

def init_tau2(m,k,init_TO_controls):
    return init_TO_controls[2][k]
        
def init_q0(m,k,init_TO_states):
    return init_TO_states[0][k]

def init_q1(m,k,init_TO_states):
    return init_TO_states[1][k]

def init_q2(m,k,init_TO_states):
    return init_TO_states[2][k]

def init_v0(m,k,init_TO_states):
    return init_TO_states[3][k]

def init_v1(m,k,init_TO_states):
    return init_TO_states[4][k]

def init_v2(m,k,init_TO_states):
    return init_TO_states[5][k]

# Warmstarting with initial conditions

def init_q0_ICS(m,k,prev_state):
    return prev_state[0]

def init_q1_ICS(m,k,prev_state):
    return prev_state[1]

def init_q2_ICS(m,k,prev_state):
    return prev_state[2]

def init_v0_ICS(m,k,prev_state):
    return prev_state[3]

def init_v1_ICS(m,k,prev_state):
    return prev_state[4]

def init_v2_ICS(m,k,prev_state):
    return prev_state[5]

# Warmstarting with zeros

def init_0(m,k):
    return 0.0

