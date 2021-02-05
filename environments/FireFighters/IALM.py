# -*- coding: utf-8 -*-

import numpy as np
import time
from environments.FireFighters.simulator import ffg_env
import pdb
from environments.FireFighters.utility import scomposition
np.random.seed(42)

class IALM (ffg_env):

    def __init__(self, parameters,  Influence, policy, verb=0):
        #n_iter number of iterations for the simulator
        #hor the horizon

        ffg_env.__init__(self, parameters)
        self.Influence=Influence
        self.policy=policy
        self.D_set=self.generate_d_set()

    def IALM_get_transitions(self, prev_s, a, s, t):

        #Input: the factor whose CPT must be computed, an array of actions a at time t, an array of states factors at time t
        #Output: the CPT of the specified factor given the current input actions and state a, s.
        
        prev_d_set, prev_x1, prev_x2 = prev_s
        d_set, x1, x2 = s
        

        if d_set[-1][0]!=x1 or prev_x2!=d_set[-1][2] or a!=d_set[-1][1]:
            #Check if the D set are compatible
                #print('sono qui 1')
                return 0
        if len(prev_d_set)>0:
            if np.array_equal(prev_d_set,d_set[:-1])==False or prev_d_set[-1][0]!=prev_x1:
                #print('sono qui 2')
                return 0


        #P(x2 | prev_x2 ,a1, prev_x1) is a scalar 
        T2=self.get_transitions(a=np.array([0,a]), prev_s=np.array([0,prev_x1,prev_x2]),fact=2)[x2]
        #print(T2)
        #input()
        #Array of prob distribution over x0 give prev_d_set. dim (I)= dim(x0) = f_level
        if t==0:
            I=self.Influence[0]
        else:
            I=self.Influence[t][self.d_set_to_index(prev_d_set,t)]
        P1=np.array([np.array([self.get_transitions(prev_s=np.array([prev_x0,prev_x1,prev_x2]), a=np.array([a0,a]),fact=1)[x1] for prev_x0 in range(self.f_lev)]) for a0 in range(2)])
        pi=np.array([self.policy([prev_x0,prev_x1]) for prev_x0 in range(self.f_lev)])
        d=np.diagonal(np.dot(pi,P1))
        T1 = np.dot(d,I)
        return T2*T1

    def IALM_get_rewards(self, s):
        
        d_set, x1, x2 = s
        return -x1-x2

    def IALM_initial_state(self):
        x1=np.random.choice(np.arange(self.f_lev),p=self.b_init[1])
        x2=np.random.choice(np.arange(self.f_lev),p=self.b_init[2])
        d_set=np.array([x1, 0, x2])
        s=np.array([d_set,x1,x2])
        return s

    def value_iteration(self):
        T=time.time()
        V=[]
        A=[]
        for t in range(self.hor,-1,-1):
            print('Value iteration time',self.hor-t)
            S, dim_S=self.state_space(t)
            V.append([])
            if t==self.hor:
                for s in S:
                    V[0].append(self.IALM_get_rewards(s))
            else:
                rev_index=self.hor-t-1
                A.append([])
                prev_V=V[rev_index]
                S_next, dim_S_next=self.state_space(t+1)
                i=0
                for prev_s in S:
                    i+=1
                    #if t==self.hor-1 and i>=dim_S/100 and i<dim_S/100+1:
                    #    print('Tempo per processare un centesimo di S al tempo hor-1', time.time()-T)
                    Qval=[self.IALM_get_rewards(prev_s)+np.sum([self.IALM_get_transitions(prev_s,a,s,t)*prev_V[j] for j,s in enumerate(S_next)]) for a in range(2)]
                    V[-1].append(np.max(Qval))
                    A[-1].append(np.argmax(Qval))
                A[-1]=np.array(A[-1])
        #initial_state=np.array([np.array([[self.f_lev-1,0,self.f_lev-1]]),self.f_lev-1,self.f_lev-1])
        #initial_index=self.state_to_index(initial_state,0)
        #print('Value in approximate IALM',np.array(V[-1])[initial_index])
        print('Time for value iteration',time.time()-T)
        return np.flip(A)


    def evaluate_IALM_pi_global_env(self, IALM_pi, policy, n):
        V=np.zeros(n)
        D_set=[]
        T=time.time()
        for i in range(n):
            s=self.initial_state()
            a=np.zeros(self.n_agents)
            for t in range(self.hor):

                o = [s[0],s[1]]
                a_distr = policy(o)
                a[0]=np.random.choice(np.arange(len(a_distr)), p=a_distr)
                
                if t==0:
                    s_IALM=np.array([s[1],s[2]])
                    index=s[1]*self.f_lev+s[2]
                else:
                    D_set.append(np.array([s[1],a[1],prev_s[2]]))
                    s_IALM=np.array([D_set ,s[1],s[2]])
                    index=self.state_to_index(s_IALM,t)
                    S,_=self.state_space(t)
                a[1]=IALM_pi[t][index]
                prev_s=np.copy(s)
                
                p=np.array([self.get_transitions(prev_s, a, k ) for k in range(self.n_fact)])

                for k in range(self.n_fact):
                    s[k] = np.random.choice(np.arange(len(p[k])), p=p[k])
            
                r=self.get_rewards(prev_s,a,s)
                if self.verb:
                    print(t,'Previous state',prev_s)
                    print (t, 'State   ', s)
                    print (t, 'Actions ', a)
                    print(t, 'Rewards', r)
                    print ('\n')
                V[i]+=r[1]
            V[i]+=self.get_rewards(s,a,s)[1]
            if self.verb:
                print ('Value', V[i])
        V=np.mean(V)
        print('Time for evaluate approximate optimal policy', time.time()-T)
        return V


    def d_set_to_index(self,d_set,t):
        step_d=self.generate_1_step_d_set()
        indexes=[]
        dim_d_set=(self.f_lev*2*self.f_lev)
        for i in range(t):
            app=d_set[i]
            for j in range(np.shape(step_d)[0]):
                if np.array_equal(step_d[j],app):
                    break
            indexes.append(j)
        index=np.sum([dim_d_set**i * np.flip(indexes)[i] for i in range(len(indexes))])
        return index


    def generate_1_step_d_set(self):
    #Generate all the possible instatiation fo one time step d_set in the binary order 0,0,0; 0,0,1...
        D_set_1_step=[]
        for i in range(self.f_lev):
            for j in range(2):
                for k in range(self.f_lev):
                    D_set_1_step.append(np.array([i,j,k]))
        return np.array(D_set_1_step)


    def index_to_dset(self, i, t):
        dim_d_set=(self.f_lev*2*self.f_lev)
        dset_index=scomposition(i,dim_d_set,[])
        while(len(dset_index))<=t:
            dset_index.insert(0,0)
        return np.array(dset_index)
        

    def generate_d_set(self):
        #Generate all the possible instantiation of the D_set
        # D set (t) to predict source influence time t (D(0) is empty) 
        T=time.time()
        D_set=[]
        dim_d_set=(self.f_lev*2*self.f_lev)
        for t in range(self.hor):
            D_set.append([])
            for i in range(dim_d_set**(t+1)):
                dset_index=self.index_to_dset(i,t)
                D_set[t].append([])
                for j in dset_index:
                    D_set[t][i].append((self.generate_1_step_d_set()[j]))
        T=time.time()-T
        D_set.insert(0,[])
        print('Time to compute all instatiations of the D_set', T)
        return np.array(D_set)

    def state_to_index(self, s , t):
        dset, x1, x2 = s
        index=self.d_set_to_index(dset,t)
        index=index*self.f_lev +x2
        return index


    def state_space(self,t):
        #State space at time step t
        S=[]
        if t==0:
            dim_S=self.f_lev**2
            for i in range(self.f_lev):
                for j in range(self.f_lev):
                    S.append(np.array([[],i,j]))
        else:
            dim_S=self.f_lev*(self.f_lev*2*self.f_lev)**t
            dim_D=len(self.D_set[t])
            for d in range(dim_D):
                    D_set=self.D_set[t][d]
                    for x2 in range(self.f_lev):
                        s=np.array([D_set,D_set[-1][0],x2])
                        S.append(s)
        return S, dim_S





   

    

