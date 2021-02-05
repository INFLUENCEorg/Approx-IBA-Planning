
import numpy as np

class agent(object):

    def __init__(self, policy):

        if policy=='rand':
            self.policy=self.rand_pi
        elif policy=='left':
            self.policy=self.pi0
        elif policy=='right':
            self.policy=self.pi1
        elif policy=='rand_left':
            self.policy=self.rand_left
        elif policy=='rand_right':
            self.policy=self.rand_right

    def rand_left(self,o):
        if o[0]>=o[1]:
            return np.array([0.8,0.2])
        else:
            return np.array([0.2,0.8])
    
    def rand_right(self,o):

        if o[1]>=o[0]:
            return np.array([0.2,0.8])
        else:
            return np.array([0.8,0.2])

    def pi0(self, o):
        if o[0]>=o[1]:
            return np.array([1,0])
        else:
            return np.array([0,1])

    def pi1(self,o):

        if o[1]>=o[0]:
            return np.array([0,1])
        else:
            return np.array([1,0])

    def rand_pi(self,o):
        return np.array([0.5, 0.5])






