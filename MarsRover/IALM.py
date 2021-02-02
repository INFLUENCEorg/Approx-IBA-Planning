# -*- coding: utf-8 -*-

import numpy as np
import time
from MarsRover import MR_env
import pdb
from utility import index_to_d_set, d_set_to_index, generate_d_set
np.random.seed(42)

class IALM (MR_env):

    def __init__(self, parameters,  Influence):

        MR_env.__init__(self, parameters)
        self.Influence=Influence

    def IALM_get_rewards(self, prev_s, a, s):
        #Output: (float) reward of the local agent
        D_prev, prev_x1, prev_x2 = prev_s
        D, x1, x2= s
        
        if x2==self.n_states:
            if prev_x2!=self.n_states:
        	    R=self.max_r
            else:
                R=0
        else: 
        	if x2==prev_x2:
        		if a==0:
        			R=0
        		else:
        			R=self.r_fail
        	else:
        		R=self.r_succ
        	
        return R


    def IALM_initial_state(self):

        x1=np.random.choice(np.arange(self.X1_dim),p=self.b_init[1])
        x2=np.random.choice(np.arange(self.X2_dim),p=self.b_init[2])
        d_set=np.array([x1])
        s=np.array([d_set,x1,x2])
        return s


    def IALM_get_transitions(self, prev_s, a, s,t):
    	#input: prev_s= (prev_D_set, prev_x1 ,prev_x2), a=a_MR, s=(D_set, x1, x2), time step t (after t time steps the process started)
        #Output: the probability T (s | prev_s, a, t)
        
        prev_d_set, prev_x1, prev_x2 = prev_s
        d_set, x1, x2= s

        if d_set[-1]!=x1 or prev_d_set[-1]!=prev_x1 or np.array_equal(d_set[:-1],prev_d_set)==False:
            return 0

        T2=self.get_transitions_pos(prev_pos=prev_x2, a_mr=a, plan=x1)
        
        I=self.Influence[t][d_set_to_index(t,prev_d_set)]

        T1=np.zeros(2)
        for fac1 in range(2):
            T1[fac1]=np.sum([self.get_transitions_plan(a_sat=a0, prev_plan=prev_x1)[fac1]*I[a0] for a0 in range(2)])
            #print(fac1)
            #print(T1[fac1])
            #print(I)
            #print(t)
            #input()

        return T2[x2]*T1[x1]

    def evaluate_policy(self, n, policy):
        #the policy is a list len(policy) = hor, where policy[t] for every index of the IALM state space when t actions have still to be picked, gives the index of an action of the MR
        # n number of simulations
        T=time.time()
        V=np.zeros(n)
        for i in range(n):
            s=self.IALM_initial_state()
            s_index=self.state_to_index(self.hor,s)
            R=0
            for t in range (self.hor):
                S_next,dim_S_next=self.state_space(self.hor-t-1)
                a=policy[self.hor-t-1][s_index]
                prev_s=np.copy(s)
                p=np.array([self.IALM_get_transitions(prev_s,a,s,t) for s in S_next])
                s_index = np.random.choice(np.arange(dim_S_next), p=p)
                s=S_next[s_index]
                R+=self.IALM_get_rewards(prev_s,a,s)
            V[i]=R
        #print('Time for policy evaluation',time.time()-T)
        return np.mean(V)
    

    def exact_evaluate_policy(self,policy):
        T=time.time()
        V=[]
        for t in range(self.hor+1):
            V.append([])
            S, dim_S=self.state_space(t)
            if t==0:
                V[0]=np.zeros(dim_S)
            else:
                prev_V=V[t-1]
                S_next, dim_S_next=self.state_space(t-1)
                for i, prev_s in enumerate(S):
                    a=policy[t-1][i]
                    Qval_a=np.sum([self.IALM_get_transitions(prev_s,a,s,self.hor-t)*(self.IALM_get_rewards(prev_s,a,s)+prev_V[j]) for j,s in enumerate(S_next)])
                    V[t].append(Qval_a)
        #print('Time for policy evaluation',time.time()-T)
        V=np.sum(np.array(V[-1])*np.array(self.IALM_b0()))
        return V

    def value_iteration(self):
    	#Output: policy, value
    	#policy is list hor x dim augmented state space for each t
    	# t stands for the actions left
        T=time.time()
        V=[]
        Pi=[]
        for t in range(self.hor+1):
            V.append([])
            S, dim_S=self.state_space(t)
            if t==0:
                V[0]=np.zeros(dim_S)
            else:
                Pi.append([])
                prev_V=V[t-1]
                S_next, dim_S_next=self.state_space(t-1)
                for prev_s in S:
                    Qval=[np.sum([self.IALM_get_transitions(prev_s,a,s,self.hor-t)*(self.IALM_get_rewards(prev_s,a,s)+prev_V[j]) for j,s in enumerate(S_next)]) for a in range(2)]
                    V[t].append(np.max(Qval))
                    Pi[t-1].append(np.argmax(Qval))
        V=np.sum(np.array(V[-1])*np.array(self.IALM_b0()))
        #print(V)
        #print(Pi)
        #input()
        #print('Time for value iteration',time.time()-T)
        return Pi, V
      
    
    def state_to_index(self,t,s):
    	# t is the number of steps/actions left
        #output index of the state according to d set binary enumeration at time t
        S, dim_S=self.state_space(t)

        for index in range(dim_S):
            if np.array_equal(S[index][0],s[0]) and S[index][-1]==s[-1]:
                return index

    def IALM_b0(self):
    	S,_=self.state_space(self.hor)
    	b0=[]
    	for s in S:
    		b0.append(self.b_init[1][s[1]]*self.b_init[2][s[2]])
    	return b0

    def state_space(self,t):
        #State space when the agent has still t actions to perform
        S=[]
        dim_S=2**(self.hor-t+1)*(self.n_states+1)

        for i in range(2**(self.hor-t+1)):
            D_set = "{0:b}".format(i)
            D_set = np.array([int(j) for j in D_set])
            while(len(D_set)<self.hor-t+1):
                D_set=np.insert(D_set,0,0,axis=0)
            for x2 in range(self.n_states+1):
                S.append([D_set,D_set[-1],x2])
        return S, dim_S





   

    

