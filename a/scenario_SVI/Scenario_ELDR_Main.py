#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:05:29 2020

@author: zcui
"""

import numpy as np
import pandas as pd
#import gurobipy as gp
#from gurobipy import *
#from gurobipy import GRB
#import time


import sys
sys.path.append(r'/Users/zcui/Documents/Research/Inventory Routing Problem/IRP_Code/Scenario_based_ELDR/Cplex')
import Scenario_ELDR_Function as IRP
import os

logTo_flag=1


N = 20
T = 6
W = 2

H= np.loadtxt("routes_n20_i5_b0.7.txt", delimiter="\t")
H=H[:-1,1:]
N=H.shape[1]
R=H.shape[0]

#%%
#---------------------------------
# setting for A B tau

A = [1,1]
B = [1,1]
tau_lb = np.zeros((N, T), dtype=np.int)
tau_ub = np.ones((N, T)) * 50



#----------------------------------------------------------------------------------------------------------
D_lb = np.zeros((W,N,T))
D_ub = np.zeros((W,N,T))

D_lb[0,:,:] = 10
D_ub[0,:,:] = 40
D_lb[1,:,:] = 10
D_ub[1,:,:] = 40


D_lb_ns = np.zeros((1,N,T))
D_ub_ns = np.zeros((1,N,T))
D_lb_ns[0,:,:]=np.minimum(D_lb[0,:,:], D_lb[1,:,:])
D_ub_ns[0,:,:]=np.maximum(D_ub[0,:,:], D_ub[1,:,:])


# Utility function:
K = 2
a = [1, 0]
b = [0, -1]

K_l = 2
o = [1, -1]
p = [0, 0]

MM = 10000

P = np.ones(W)/W

#-------------------------------------------------------------------------
# training data
N_train=[int(P[w]*10000) for w in range(W)]
E1, E2, e1, e2, E1_ns, E2_ns, e1_ns, e2_ns = IRP.Moment_ambiguity_set_parameter_generation(N_train, W, N, T, A, B, D_ub, D_lb)

#-------------------------------------------------------------------------
# testing data
N_test=[int(P[w]*1000) for w in range(W)]
N_test_total = sum(N_test)

D_test={}
for w in range(W):
    Dw_test = np.zeros((N,T,N_test[w]))
    for i in range(N_test[w]):
        Dw_test[:,:,i] = np.random.beta(A[w], B[w], [N, T])*(D_ub[w,:,:] - D_lb[w,:,:])+D_lb[w,:,:]
    D_test[w] = Dw_test


#----------------------------------------------------------------
# valid_inequality_parameter
valid_inequality_parameter = IRP.valid_inequality_calculator(N,T,W,P,K,a,b,tau_lb,tau_ub,D_lb,D_ub,E1,E2)


#%
#----------------------------------------------------------------
# performance dataframe

perf={}
perf['Model']=[]
perf['Objective']=[]
perf['Probability']=[]
perf['Expectation']=[]
perf['Time']=[]



#----------------------------------------------------------------------------------------------------------------
#'Scenario LDR - valid - Benders'
#----------------------------------------------------------------------------------------------------------------

obje_LvB,alpha_t_LvB,y_t_LvB,q_t_LvB,x_t_LvB,v_t_LvB,p_t_LvB,solve_time_LvB\
    = IRP.Scenario_ELDR_calculate(\
        W,N,T,R,P,H,K,a,b,K_l,o,p,tau_lb,tau_ub,D_lb,D_ub,E1,E2,e1,e2,\
        MM,N_test,D_test,logTo_flag,valid_inequality_parameter,valid_flag=True,benders_flag=True)
        
perf['Model'] = perf['Model']+['Scenario LDR - valid - Benders']
perf['Objective'] = perf['Objective']+[obje_LvB]
perf['Probability'] = perf['Probability']+[sum(np.mean(p_t_LvB[w])*P[w] for w in range(W))]
perf['Expectation'] = perf['Expectation']+[sum(np.mean(v_t_LvB[w])*P[w] for w in range(W))]
perf['Time'] = perf['Time']+[solve_time_LvB]





perf_df = pd.DataFrame(perf)
print(perf_df)

perf_df.to_csv("N%s_T%s_R%s_W%s_I%s.csv" % (N,T,R,W),float_format='%.4f')

