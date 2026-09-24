#%% open in VScode jupyter notebook and run the code in cells

import numpy as np
import time

from scipy.special import softmax
import torch
import torch.nn.functional as F
#%% reinforcement learning basics 
# code written based on 01_RLBasics.docx -> Basic->RL intro


#experimental setup
#state = position 
#action=1 , go right , action=0 then go left
#normal state? reward -1 , at the destination(state=position4)?, then reward =10



#this environment class creates the environment .position from 0 to 4 for now

class SimpleEnvironment:

    def __init__(self) :
        self.position=0 ; 
    def reset (self):
        self.position = 0 
        return self.position 
    def step(self,action):

        if (action==1): # go right
            self.position+=1
        elif action==0: # go left
            self.position-=1 

            # take the position between 0 and 4
        self.position =  min(4,self.position)
        self.position  =max(0,self.position)

        #position 4 is the destiny .reward maximum
        if self.position==4: 
            reward=10 
            done=True 
            # normal state ..negative reward 
        else :
            reward =-1 
            done=False 
        return self.position,reward, done 


env = SimpleEnvironment() #creating environment object
state = env.reset()
print("initial state=",state)

#only random action and change the state .No learning from reward for now
for t in range(100):
    action = np.random.choice([0,1]) 
    next_state, reward, done = env.step(action)

    print(
        "state =", state,
        "action =", action,
        "reward =", reward,
        "next_state =", next_state
    )
    state = next_state
    if done:
        break
    time.sleep(.5)


print(f"done.total iterations={t}")




#%%  Policy — π(a∣s) 

# state - neural network- z=logits - softmax - probs 

logits = torch .tensor ([2.0,1.0,.1,.2,.3,3]) 
probs = F.softmax(logits,dim=0 ) 
probs 


#stochastic action selection .
#multinomial samples one action based on probability distribution.it doesnt mean multinomial always take highest probability action
#rather it may sometimes samples medium range probability action as well . .
# but that comes in so leff frequently than the highest probability action . 


action = torch.multinomial(probs,1) #stochastic action  
action.item() 


#if we take greedy action , just take the highest .not recommened . 
greedy_action = torch.argmax(probs)
greedy_action.item() 



#%% RL lesson3 Value functions 

# we are given rewards for each state in an episode . we want to calculate the return Gt for each state in the episode .

#lets find G when for state 0 

#iterative formula G_t = Rt+1 + gamma*G_t+1


gamma = .9 #discount factor
rewards =[-1,-1,-1,10]  #rewards for 0->1->2->3->4 

G =0 #initilize reward for G_3

reversed(rewards)  #reverse the rewards to calculate Gt from the end of the episode

for reward in  reversed(rewards):
    G = reward + gamma*G 
    


print(G) # G for state 0  . 

#whats happening?

# G3 = 0 R4=10 
# find G2 
# now R3=-1 , find G1 
# Now R2=-1 , find G0
# we have gone in reverse order to find G0 from G3 .



#%% usisng bell man 

# we have S0 to S4.assume only policy is to go right 
# we need to find Value function V at each state using bellman equation Vπ(s)=Eπ​[Rt+1​+γVπ(St+1​)∣St​=s]

gamma=.9 

V = np.zeros(5)
V[4] =0 # at the goal=state4
R= np.array([-1,-1,-1,-1,10])  # rewars of all the state  (not return..return G, reward =R)

# starting from state=3 . 
for i in reversed(range(1,5) ): 
  
  V[i-1] = R[i]+ gamma  * V[i] # bellman equation
  print(f'state{i-1} V={V[i-1]}' ) 

  time.sleep(.1)
    
 #output       
#state3 V=10.0
#state2 V=8.0
#state1 V=6.2
#state0 V=4.58



#%%   implementing (I-γP^π ) V^π=r^π. 
#  our states are s0 s1 s2 s3 s4(goal)
#right =80% , left =20% 
#lets make it

gamma =.9  # discount factor determines how much future we want to consider 

P_pi = np.array([
    [0.2, 0.8, 0.0, 0.0],  # From s0
    [0.2, 0.0, 0.8, 0.0],  # From s1
    [0.0, 0.2, 0.0, 0.8],  # From s2
    [0.0, 0.0, 0.2, 0.0]   # From s3; 0.8 goes to terminal
])

# Expected immediate reward under policy pi
r_pi = np.array([
    -1.2,  # Expected reward from s0
    -1.0,  # Expected reward from s1
    -1.0,  # Expected reward from s2
     7.8   # Expected reward from s3
])

I = np.eye(4) # Identity matrix for 4 states (s0 to s3)
I

#the equation is (I - γP^π)V^π = r^π   
A = I - gamma * P_pi  # (I - γP^π) 

V_pi  = np.linalg.solve(A, r_pi)  # Solve for V^π 

print(V_pi.reshape(-1,1)  ) 




#%% iterative process 
import numpy as np
from time import sleep
gamma = 0.9 

# Policy-induced transition matrix P^pi
# Rows = current states [s0, s1, s2, s3]
# Columns = next states [s0, s1, s2, s3]
P_pi = np.array([
    [0.2, 0.8, 0.0, 0.0],  # From s0
    [0.2, 0.0, 0.8, 0.0],  # From s1
    [0.0, 0.2, 0.0, 0.8],  # From s2
    [0.0, 0.0, 0.2, 0.0]   # From s3
])

# Expected immediate reward under policy pi
r_pi = np.array([
    -1.2,
    -1.0,
    -1.0,
     7.8
])

V = np.zeros(4)
V_new =np.zeros(4) 

pivot = 1e-20

for k in range(100): 
    V_new = r_pi + P_pi @ V 
    print (f"iteration={k} V =  {V_new}" )

    if abs(np.max( V-V_new  )  )<pivot:
        break



    V = V_new 
    # sleep(.1)




