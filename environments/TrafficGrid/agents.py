
import numpy as np

class traffic_agent(object):

    def __init__(self, policy):

        if policy=='rand':
            self.policy=self.rand_pi

        if policy=='w':
            self.policy=self.policy_west

        if policy=='e':
            self.policy=self.policy_east

        if policy=='n':
            self.policy=self.policy_north

        if policy=='s':
            self.policy=self.policy_south

    def rand_pi(self, local_s):
        s,e,n,w = local_s
        action_distr=np.array([0.5,0.5])
        return action_distr

    def policy_west(self,local_s):
        s,e,n,w = local_s
        if w!=0:
            action_distr=[1.,0.]
        else:
            action_distr=[0.,1.]
        return action_distr

    def policy_north(self,local_s):
        s,e,n,w = local_s
        if n!=0:
            action_distr=[0.,1.]
        else:
            action_distr=[1.,0.]
        return action_distr

    def policy_east(self,local_s):
        s,e,n,w = local_s
        if e!=0:
            action_distr=[1.,0.]
        else:
            action_distr=[0.,1.]
        return action_distr
    
    def policy_south(self,local_s):
        s,e,n,w = local_s
        if s!=0:
            action_distr=[0.,1.]
        else:
            action_distr=[1.,0.]
        return action_distr

