# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:08:42 2021

@author: Ambrosio Valencia
"""

import numpy as np
from itertools import product
import quantecon.game_theory as gt

# import strategy_dynamics from as sdyn
from strategy_dynamics import powergset, nx2gameset, nx2subgames, nx2feargreed_i

n = 4

pset, rset = powergset(range(3))
gset, N__i = nx2gameset(n,1,start_from=1)


for J in range(len(pset)):
    print([pset[J], rset[J]])
print('-------------------------------------------')
for g in range(nx2subgames(n)):
    print(gset[g])
print('-------------------------------------------')




#%% Test with 3 players

p = { k:gt.Player(np.reshape(np.arange(8), [2,2,2]) + 8*k) for k in range(3) }

nf = gt.NormalFormGame((p[0],p[1],p[2]))

u000 = np.array([    3,     0,     2])
u010 = np.array([    0,     2,     0])
u100 = np.array([    0,     1,     0])
u110 = np.array([    1,     0,     0])
u001 = np.array([    1,     0,     0])
u011 = np.array([    0,     1,     0])
u101 = np.array([    0,     3,     0])
u111 = np.array([    2,     0,     3])

nf[0,0,0] = u000; nf[0,0,1] = u001; nf[0,1,0] = u010; nf[0,1,1] = u011
nf[1,0,0] = u100; nf[1,0,1] = u101; nf[1,1,0] = u110; nf[1,1,1] = u111

arr = np.array( list( product( *3*(range(2),) ) ) )

# for i in range(3):
#     for idc in arr:
#         print(str(idc)+': '+'{:2d}'.format(nf[idc][i])+' --> '+\
#               str(np.array(nx2shiftindices(idc, i)))+': '+\
#               '{:2d}'.format(nf.players[i].payoff_array[nx2shiftindices(idc, i)]))
#     print('---------------------------')
    
for i in range(3):
    Fi, Gi, Hi = nx2feargreed_i(i, nf.players[i].payoff_array)
    print(np.vstack((Fi,Gi)).T)
    print('Harmony index: '+str(Hi))
    print('---------------------------')
    
#%% Test with 4 players

n_ = 4
# P_ = [ gt.Player(np.zeros(n_*(2,))) for k in range(n_) ]
rand_arr = np.random.rand(*n_*(2,))
P_ = [ gt.Player(rand_arr) for k in range(n_) ] # This game is symmetric
nf_ = gt.NormalFormGame(P_)

# nf_[0,1,1,1] =  1.1915317282104736*np.ones(n_)
# nf_[1,0,1,1] = -0.5791282258424457*np.ones(n_)
# nf_[1,1,0,1] =  2.2908030291300454*np.ones(n_)
# nf_[1,1,1,0] =  1.7981231853465840*np.ones(n_)
# nf_[1,1,1,1] =  1.6822035378084048*np.ones(n_)

for i in range(n_):
    # print(nf_.players[i].payoff_array)
    Fi, Gi, Hi = nx2feargreed_i(i, nf_.players[i].payoff_array)#, shift=True)
    print(np.vstack((Fi,Gi)).T)
    print('Harmony index: '+str(Hi))
    print('---------------------------')
