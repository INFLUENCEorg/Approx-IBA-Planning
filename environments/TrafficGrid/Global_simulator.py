# -*- coding: utf-8 -*-

import numpy as np
import time
from environments.TrafficGrid.utility import index_to_d_set, d_set_to_index,  generate_d_set
import pylab

np.random.seed(42)

class Traffic_env (object):

    def __init__(self, parameters,verb=0):

        #General parameters
        self.parameters=parameters
        self.dim_lane=self.parameters['dim_lane']
        self.n_lanes=4
        self.n_agents=4

        #Transition parameters
        self.p_out_in= self.parameters['p_out_in']
        self.p_in=self.parameters['p_in']  
        
        self.cross0_out, self.cross1_out=int(self.dim_lane/4), int(3*self.dim_lane/4)
        
        self.b0=self.parameters['b0']
        #Simulation parameters
        self.hor=self.parameters['hor']
        self.trained=None
        self.n_iter=self.parameters['n_iter']
        self.verb=verb

    def initial_state(self):
        #Output: sample an initial state according to the initial distribution b_init 
        if self.b0 is None:
            l0=np.zeros(self.dim_lane)
            l1=np.zeros(self.dim_lane)
            l2=np.zeros(self.dim_lane)
            l3=np.zeros(self.dim_lane)
        else:
            l0,l1,l2,l3=self.b0

        s=np.array([l0,l1,l2,l3])
        return s

    def get_rewards(self, s):
        #Output: float reward of Mars Rover
        #Input: previous state tuple prev_s=(x0,x1,x2), tuple of actions a=(satellite action, MR action), tuple next state s=(x0',x1',x2'). 
        l0,l1,l2,l3=s
        local_s=self.get_local_state(s)
        R=-np.sum(local_s[0][np.array([1,2])])
        return R

    def sample_s(self,prev_s,a):

        prev_l0,prev_l1,prev_l2,prev_l3 = prev_s
        a0,a1,a2,a3=a

        l0=self.sample_l(prev_l0,np.array([a3,a0]),pos='ver')
        l1=self.sample_l(prev_l1,np.array([a1,a2]),pos='ver',incoming=prev_l0[-1])
        l2=self.sample_l(prev_l2,np.array([a1,a0]),pos='hor')
        l3=self.sample_l(prev_l3,np.array([a3,a2]),pos='hor',incoming=prev_l2[-1])

        s=np.array([l0,l1,l2,l3])

        return s
    
    
    def sample_l(self,prev_l, a_pair, pos, incoming=None):
        l=np.zeros(self.dim_lane,dtype=int)
        a0, a1 = a_pair

        l[self.cross1_out+1:]=prev_l[self.cross1_out:-1]
        force_red=0

        if pos=='hor':
            a0, a1 = 1-a0,1-a1
        
        if incoming:
            p_in=self.p_in+ self.p_out_in -self.p_in * self.p_out_in
        else:
            p_in=self.p_in

        if a1==1:
        #green to the lane 
            l[self.cross0_out+1:self.cross1_out+1]=prev_l[self.cross0_out:self.cross1_out]
        else:
        #red to the lane 
            l[self.cross1_out]=0
            for i in np.arange(self.cross1_out-1,self.cross0_out-1,step=-1):
                if prev_l[i]!=1:
                    break
            if i==self.cross0_out and prev_l[i]==1:
                l[self.cross0_out:self.cross1_out]=prev_l[self.cross0_out:self.cross1_out]
                a0=0
                force_red=1
            elif i==self.cross0_out:
                l[i+1:self.cross1_out]=prev_l[i+1:self.cross1_out]
            else:
                l[i+1:self.cross1_out]=prev_l[i+1:self.cross1_out]
                l[self.cross0_out+1:i+1]=prev_l[self.cross0_out:i]

        if a0==1:
        #green to the lane 
            l[1:self.cross0_out+1]=prev_l[0:self.cross0_out]
            l[0]=np.random.choice([0,1],p=[1-p_in,p_in])
        else:
        # red to the lane
            if force_red==0:
                l[self.cross0_out]=0
            for i in np.arange(self.cross0_out-1,-1,step=-1):
                if prev_l[i]!=1:
                    break
            l[i+1:self.cross0_out]=prev_l[i+1:self.cross0_out]
            l[1:i+1]=prev_l[0:i]
            if i!=0:
                l[0]=np.random.choice([0,1],p=[1-p_in,p_in])
            else:
                l[0]=1

        return l

    def plot_state(self,s,a):

        l0,l1,l2,l3=s
        a0,a1,a2,a3=a
        
        colors=[]        
        traffic_lights=[]
        for i in range(4):
            colors.append(np.insert(s[i],[self.cross0_out, self.cross1_out],[0,0]))
            colors[-1]=np.where(colors[-1]==0, 'w', colors[-1])
            colors[-1]=np.where(colors[-1]!='w', 'b', colors[-1])      
            if a[i]==0:
                traffic_lights.append('_')
            else:
                traffic_lights.append('|')
        
        x0,x1,y0,y1=0.5+self.cross0_out, 0.5+self.cross1_out+1, 0.5+self.cross0_out, 0.5+self.cross1_out+1
        coor=[self.cross0_out,self.cross0_out+1, self.cross1_out+1,self.cross1_out+2]

        dim_grid=self.dim_lane+2

        fig = pylab.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, dim_grid, 1))
        ax.set_yticks(np.arange(0, dim_grid, 1))
        pylab.xlim(0,dim_grid)
        pylab.ylim(0,dim_grid)
        for x in coor:
            pylab.axvline(x=x, c='k')
            pylab.axhline(y=x, c='k')

        #pylab.title('State '+str(s)+'\n Action '+str(a))

        pylab.scatter(np.ones(dim_grid)* x0,np.arange(dim_grid-1+0.5,-1+0.5,step=-1), c=colors[0], marker='v',s=100, label='lane0')
        pylab.scatter(np.ones(dim_grid)*x1,np.arange(0.5,dim_grid+0.5,step=1), c=colors[1], marker='^',s=100,label='lane1')
        pylab.scatter(np.arange(dim_grid-1+0.5,-1+0.5,step=-1),np.ones(dim_grid)*y0, c=colors[2], marker='<',s=100, label='lane2')
        pylab.scatter(np.arange(0.5,dim_grid+0.5,step=1),np.ones(dim_grid)*y1, c=colors[3], marker='>',s=100,label='lane3')


        pylab.scatter(x0,y0,marker=traffic_lights[0],s=500, c='g')
        pylab.scatter(x1,y0,marker=traffic_lights[1],s=500, c='g')
        pylab.scatter(x1,y1,marker=traffic_lights[2],s=500, c='g')
        pylab.scatter(x0,y1,marker=traffic_lights[3],s=500, c='g')

        #pylab.gca().set_aspect("equal")
        pylab.grid()
        pylab.savefig('grid.pdf')
        pylab.show()
        #pylab.draw()
        #pylab.pause(0.1)



    def get_local_state(self,s):
        l0,l1,l2,l3=s
        local0=np.array([l0[self.cross1_out],l2[self.cross1_out-1],l0[self.cross1_out-1],l2[self.cross1_out]])
        local1=np.array([l1[self.cross0_out-1],l2[self.cross0_out-1],l1[self.cross0_out],l2[self.cross0_out]])
        local2=np.array([l1[self.cross1_out-1],l3[self.cross1_out],l1[self.cross1_out],l3[self.cross1_out-1]])
        local3=np.array([l0[self.cross0_out],l3[self.cross0_out],l0[self.cross0_out-1],l3[self.cross0_out-1]])

        local_s=np.array([local0,local1,local2,local3])

        return local_s
    
    def plot_local_state(self,local_s,a):
        
        colors=[]        
        traffic_light=[]
        
        pos_x=np.array([0.5+1,0.5+2,0.5+1,0.5])
        pos_y=np.array([0.5,0.5+1,0.5+2,0.5+1])
   
        colors=np.where(local_s==0, 'w', local_s)
        colors=np.where(colors!='w', 'b', colors)  
        
        if a==0:
            traffic_light='_'
        else:
            traffic_light='|'

        dim_grid=3

        fig = pylab.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, dim_grid, 1))
        ax.set_yticks(np.arange(0, dim_grid, 1))
        pylab.xlim(0,dim_grid)
        pylab.ylim(0,dim_grid)
        for x in range(1,3):
            pylab.axvline(x=x, c='k')
            pylab.axhline(y=x, c='k')
        #pylab.title('Local state '+str(local_s)+'\n Action '+str(a))
        pylab.scatter(pos_x[np.array([0,2])],pos_y[np.array([0,2])], c=colors[np.array([0,2])], marker='v',s=100)
        pylab.scatter(pos_x[np.array([1,3])],pos_y[np.array([1,3])], c=colors[np.array([1,3])], marker='<',s=100)
        pylab.scatter(1+0.5,1+0.5,marker=traffic_light,s=500, c='g')

        #pylab.gca().set_aspect("equal")
        pylab.grid()
        #pylab.show()
        #pylab.draw()
        #pylab.pause(0.1)


    def run_simulation(self, policies):
        #Input: list of policies of the two agents a0=satellite, a1=MR
        #Output: save arrays for the visited states S, action performed A, rewards received R and cumulative reward V
        # dim(S) = n_factors x n_iterations x hor
        # dim(A) = n_agents x n_iterations x hor
        # dim(R) = n_iterations x hor (only MR receives the reward
        # dim(V) = n_iteration
        T=time.time()

        self.S=np.zeros((self.n_iter, self.hor+1,self.n_lanes, self.dim_lane))
        self.A=np.zeros((self.n_iter, self.hor+1,self.n_agents))
        self.R=np.zeros((self.n_iter, self.hor+1))
        self.V=np.zeros((self.n_iter))

        for i in range(self.n_iter):

            s=self.initial_state()
            a=np.zeros(self.n_agents)
            a_distr= [None] * self.n_agents
            
            for t in range(self.hor+1):
 
                local_s=self.get_local_state(s)
                for k , pi in enumerate(policies):
                    a_distr[k] = pi(local_s[k])
                    a[k]=np.random.choice(np.arange(len(a_distr[k])), p=a_distr[k])

                if t==0:
                    a[0]=1
                elif t==1:
                    a[0]=0
                elif t==2:
                    a[0]=0

                self.S[i, t] = s
                self.A[i, t] = a   
                prev_s=np.copy(s)
                
                s=self.sample_s(prev_s, a)
                r=self.get_rewards(prev_s)

                self.R[i, t]=r

                if self.verb:
                    self.plot_state(prev_s,a)
                    local_s=self.get_local_state(prev_s)
                    self.plot_local_state(local_s[0],a[0])
                    #print(t,'Previous state',prev_s)
                    #print (t, 'State   ', s)
                    #print (t, 'Actions ', a)
                    #print(t, 'Rewards', r)
                    #print ('\n')

                self.V[i]+=r
            #pylab.show(block=True)
        T=time.time()-T
        self.trained=True
        

        print ('\nRunning time ', T)



    def extract_d_set (self):
    #Output: generated d_sets during the simulation dim(D_set) = n_iter x hor 
        D_set=[]
        for i in range(self.n_iter):
            D_set.append([])
            for j in range(self.hor):
                local_s0=self.get_local_state(self.S[i,j])[0]
                D_set[i].append(local_s0[np.array([0,3])])
        return D_set


    def extract_sources_influence(self):
        #Output: generated sources of influence during simulation dim(s_inf) =  n_iter
        s_inf=[]
        for i in range(self.n_iter):
            s_inf.append([])
            for j in range(1,self.hor+1):
                local_s0=self.get_local_state(self.S[i,j])[0]
                s_inf[i].append(local_s0[np.array([1,2])])
        return s_inf

    








   

    

