import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math

#source https://github.com/nekitboy/simulated_annealing_torch/blob/master/Task1.ipynb

class SimulatedAnnealing(Optimizer):
    def __init__(self, params, sampler, t0=1, anneal_rate=0.001,
                 min_temp=1e-5, anneal_every=100):
        defaults = dict(sampler=sampler, t0=t0, t=t0, anneal_rate=anneal_rate,
                        min_temp=min_temp, anneal_every=anneal_every, iteration=0, toprint=True)
        super(SimulatedAnnealing, self).__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise Exception("loss closure is required")

        loss = closure()

        for group in self.param_groups:
            sampler = group['sampler']

            cloned_params = [p.clone() for p in group['params']]

            for p in group['params']:
                if group['iteration'] > 0 \
                   and group['iteration'] % group['anneal_every'] == 0:
                
                    rate = -group['anneal_rate'] * group['iteration']
                    group['t'] = np.maximum(group['t0'] * np.exp(rate), group['min_temp'])

                random_perturbation = group['sampler'].sample(p.data.size())
                p.data = p.data / torch.norm(p.data)
                p.data.add_(random_perturbation)
                group['iteration'] += 1

            
            new_loss = closure()
            final_loss, is_accept, topr = self.anneal(loss, new_loss, group['t'], group['toprint'], group['min_temp'])
            group['toprint'] = topr
            if not is_accept:
                for p, prev_p in zip(group['params'], cloned_params):
                    p.data = prev_p.data

            return final_loss


    def anneal(self, loss, new_loss, t, toprint, min_t):
        def acceptance_prob(old, new, temp):
            return torch.exp((old - new)/temp)

        topr = toprint
        
        loss_v = loss.item()
        new_loss_v = new_loss.item()

        if new_loss_v < loss_v:
            return new_loss, True, topr
        else:
            # evaluate the metropolis criterion
            ap = acceptance_prob(loss, new_loss, t)
            ap_v = ap.item()
            if t == min_t and topr:
                print("old = ", loss_v, "| pert = ", new_loss_v, " | ap = ", ap_v, " | t = ", t)
                topr = False
            if ap_v > np.random.rand():
                return new_loss, True, topr

            # return the original loss if above fails
            # or if the temp is now annealed
            return loss, False, topr
        
class GaussianSampler(object):
    def __init__(self, mu, sigma, dtype='float', cuda=False):
        self.sigma = sigma
        self.mu = mu
        self.cuda = cuda
        self.dtype_str = dtype
        dtypes = {
            'float': torch.cuda.FloatTensor if cuda else torch.FloatTensor,
            'int': torch.cuda.IntTensor if cuda else torch.IntTensor,
            'long': torch.cuda.LongTensor if cuda else torch.LongTensor
        }
        self.dtype = dtypes[dtype]

    def sample(self, size):
        rand_float = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        rand_block = rand_float(*size).normal_(self.mu, self.sigma)

        if self.dtype_str == 'int' or self.dtype_str == 'long':
            rand_block = rand_block.type(self.dtype)

        return rand_block
     
