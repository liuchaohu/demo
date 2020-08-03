# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:07:29 2020


"""
import numpy as np
import gurobipy as gp
# from gurobipy import *
from gurobipy import GRB
import time
import cplex


def f(x, o, p):
    return max(o[0] * x + p[0], o[1] * x + p[1])


def e(N, M, n, m):
    c = np.zeros((N, M))
    c[n, m] = 1
    return c


def unit_vector(M, m):
    c = np.zeros(M)
    c[m] = 1
    return c


def Moment_ambiguity_set_parameter_generation(N_train, W, N, T, A, B, D_ub, D_lb):
    N_train_total = sum(N_train)

    D = {}
    for w in range(W):
        Dw = np.zeros((N, T, N_train[w]))
        for i in range(N_train[w]):
            Dw[:, :, i] = np.random.beta(A[w], B[w], [N, T]) * (D_ub[w, :, :] - D_lb[w, :, :]) + D_lb[w, :, :]
        D[w] = Dw

    E1 = np.zeros((W, N, T))
    E2 = np.zeros((W, N, T))
    e1 = np.zeros((W, T))
    e2 = np.zeros((W, N))

    for w in range(W):
        for i in range(N_train[w]):
            for n in range(N):
                for t in range(T):
                    E1[w, n, t] += D[w][n, t, i] / N_train[w]
    for w in range(W):
        for i in range(N_train[w]):
            for n in range(N):
                for t in range(T):
                    E2[w, n, t] += abs(D[w][n, t, i] - E1[w, n, t]) / N_train[w]
    for w in range(W):
        for i in range(N_train[w]):
            for t in range(T):
                e1[w, t] += abs(sum((D[w][n, t, i] - E1[w, n, t]) / E2[w, n, t] for n in range(N))) / N_train[w]
            for n in range(N):
                e2[w, n] += abs(sum((D[w][n, t, i] - E1[w, n, t]) / E2[w, n, t] for t in range(T))) / N_train[w]

    E1_ns = np.zeros((1, N, T))
    E2_ns = np.zeros((1, N, T))
    e1_ns = np.zeros((1, T))
    e2_ns = np.zeros((1, N))

    for w in range(W):
        for i in range(N_train[w]):
            for n in range(N):
                for t in range(T):
                    E1_ns[0, n, t] += D[w][n, t, i] / N_train_total
    for w in range(W):
        for i in range(N_train[w]):
            for n in range(N):
                for t in range(T):
                    E2_ns[0, n, t] += abs(D[w][n, t, i] - E1_ns[0, n, t]) / N_train_total
    for w in range(W):
        for i in range(N_train[w]):
            for t in range(T):
                e1_ns[0, t] += abs(
                    sum((D[w][n, t, i] - E1_ns[0, n, t]) / E2_ns[0, n, t] for n in range(N))) / N_train_total
            for n in range(N):
                e2_ns[0, n] += abs(
                    sum((D[w][n, t, i] - E1_ns[0, n, t]) / E2_ns[0, n, t] for t in range(T))) / N_train_total

    return E1, E2, e1, e2, E1_ns, E2_ns, e1_ns, e2_ns


def single_node_calculator(T, W, P, K, a, b, tau_lb, tau_ub, D_lb, D_ub, E1, E2):
    # M = gp.Model()
    # M.Params.LogToConsole=0

    M = cplex.Cplex()
    M.objective.set_sense(M.objective.sense.minimize)

    # alpha = M.addVars(T,lb = 0.0001,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="alpha")
    M.variables.add(obj=[1] * T,
                    lb=[0.0001] * T,
                    ub=[cplex.infinity] * T,
                    types='C' * T,
                    names=['alpha_%s' % (i) for i in range(T)])

    # q0 = M.addVars(W,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="q0")
    M.variables.add(obj=[0] * W,
                    lb=[0] * W,
                    ub=[cplex.infinity] * W,
                    types='C' * W,
                    names=['q0_%s' % (i) for i in range(W)])

    # s0 = M.addVars(W,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="s0")
    M.variables.add(obj=[0] * W * T,
                    lb=[-cplex.infinity] * W * T,
                    ub=[cplex.infinity] * W * T,
                    types='C' * W * T,
                    names=['s0_%s_%s' % (i, j) for i in range(W) for j in range(T)])

    # SS = M.addVars(W,T,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="S")#为了和前面的S区分，命名为SS
    M.variables.add(obj=[0] * W * T * T,
                    lb=[-cplex.infinity] * W * T * T,
                    ub=[cplex.infinity] * W * T * T,
                    types='C' * W * T * T,
                    names=['SS_%s_%s_%s' % (i, j, k) for i in range(W) for j in range(T) for k in range(T)])

    # TT = M.addVars(W,T,T,vtype=GRB.CONTINUOUS,name="TT")#为了和前面的T区分，命名为TT
    M.variables.add(obj=[0] * W * T * T,
                    lb=[0] * W * T * T,
                    ub=[cplex.infinity] * W * T * T,
                    types='C' * W * T * T,
                    names=['TT_%s_%s_%s' % (i, j, k) for i in range(W) for j in range(T) for k in range(T)])

    # G1_ub = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="G1_ub")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['G1_ub_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # G2_ub = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="G2_ub")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['G2_ub_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # F1_ub = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="F1_ub")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['F1_ub_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # F2_ub = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="F2_ub")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['F2_ub_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # G1_lb = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="G1_lb")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['G1_lb_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # G2_lb = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="G2_lb")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['G2_lb_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # F1_lb = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="F1_lb")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['F1_lb_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # F2_lb = M.addVars(W,T,K,T,vtype=GRB.CONTINUOUS,name="F2_lb")
    M.variables.add(obj=[0] * W * T * K * T,
                    lb=[0] * W * T * K * T,
                    ub=[cplex.infinity] * W * T * K * T,
                    types='C' * W * T * K * T,
                    names=['F2_lb_%s_%s_%s_%s' % (i, j, k, m) for i in range(W) for j in range(T) for k in range(K)
                           for m in range(T)])

    # M.setObjective(gp.quicksum(alpha[t] for t in range(T)),GRB.MINIMIZE)

    # M.addConstrs((sum(P[w]*(s0[w,t] + \
    #               sum(SS[w,t,tau]*E1[w,tau] for tau in range(T)) + \
    #               sum(TT[w,t,tau]*E2[w,tau] for tau in range(T))) for w in range(W)) \
    #               <= 0 \
    #               for t in range(T)),\
    #     name='ca0')
    rows = []
    my_rownames = ['ca0_%s' % (i) for i in range(T)]
    for t in range(T):
        row = [['s0_%s_%s' % (w, t) for w in range(W)] + ['SS_%s_%s_%s' % (w, t, tau) for tau in range(T) for w in
                                                          range(W)] + [
                   'TT_%s_%s_%s' % (w, t, tau) for tau in range(T) for w in range(W)],
               [float(P[w]) for w in range(W)] + [float(P[w] * E1[w, tau]) for tau in range(T) for w in range(W)] +
               [float(P[w] * E2[w, tau]) for tau in range(T) for w in range(W)]]
        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='L' * T,
                             rhs=[0] * T,
                             names=my_rownames)

    ##########--------------
    # M.addConstrs((sum(G1_lb[w,t,k,tau]*D_lb[w,tau] for tau in range(T)) - \
    #               sum(G2_lb[w,t,k,tau]*D_ub[w,tau] for tau in range(T)) + \
    #               sum((F2_lb[w,t,k,tau]-F1_lb[w,t,k,tau])*E1[w,tau] for tau in range(T)) \
    #               >= a[k]*(tau_lb[t]-q0[w])+b[k]*alpha[t]-s0[w,t] \
    #               for w in range(W) for k in range(K) for t in range(T)), \
    #     name='cb0')
    rows = []
    my_rownames = ['cb0_%s_%s_%s' % (w, k, t) for w in range(W) for k in range(K) for t in range(T)]
    for w in range(W):
        for k in range(K):
            for t in range(T):
                row = [['G1_lb_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)] +
                       ['G2_lb_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)] +
                       ['F2_lb_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)] +
                       ['F1_lb_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)]
                       + ['q0_%s' % (w)] + ['alpha_%s' % (t)] + ['s0_%s_%s' % (w, t)],

                       [float(D_lb[w, tau]) for tau in range(T)] + [float(-D_ub[w, tau]) for tau in range(T)]
                       + [float(E1[w, tau]) for tau in range(T)] + [float(-E1[w, tau]) for tau in range(T)]
                       + [a[k]] + [-b[k]] + [1]]
                rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='G' * W * K * T,
                             rhs=[a[k] * (float(tau_lb[t])) for w in range(W) for k in range(K) for t in range(T)],
                             names=my_rownames)

    # M.addConstrs((G1_lb[w,t,k,tau]- G2_lb[w,t,k,tau]-F1_lb[w,t,k,tau]+F2_lb[w,t,k,tau] \
    #               == (SS[w,t,tau] - a[k]*sum(unit_vector(T,m)[tau] for m in range(t+1))) \
    #               for w in range(W) for t in range(T) for k in range(K) for tau in range(T)), \
    #     name='cc0')
    rows = []
    my_rownames = ['cc0_%s_%s_%s_%s' % (w, t, k, tau) for w in range(W) for t in range(T) for k in range(K) for tau in
                   range(T)]
    for w in range(W):
        for t in range(T):
            for k in range(K):
                for tau in range(T):
                    row = [['G1_lb_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['G2_lb_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['F1_lb_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['F2_lb_%s_%s_%s_%s' % (w, t, k, tau)]
                           + ['SS_%s_%s_%s' % (w, t, tau)],

                           [1] + [-1]
                           + [-1] + [1]
                           + [-1]]
                    rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * T * K * T,
                             rhs=[-a[k] * sum(unit_vector(T, m)[tau] for m in range(t + 1)) for w in range(W) for t in
                                  range(T) for k in range(K) for tau in range(T)],
                             names=my_rownames)

    # M.addConstrs((F1_lb[w,t,k,tau] + F2_lb[w,t,k,tau] \
    #               == TT[w,t,tau] \
    #               for w in range(W)  for t in range(T) for k in range(K) for tau in range(T)),\
    #     name='cd0')
    rows = []
    my_rownames = ['cd0_%s_%s_%s_%s' % (w, t, k, tau) for w in range(W) for t in range(T) for k in range(K) for tau in
                   range(T)]
    for w in range(W):
        for t in range(T):
            for k in range(K):
                for tau in range(T):
                    row = [['F1_lb_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['F2_lb_%s_%s_%s_%s' % (w, t, k, tau)]
                           + ['TT_%s_%s_%s' % (w, t, tau)],

                           [1] + [1]
                           + [-1]]
                    rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * T * K * T,
                             rhs=[0] * W * T * K * T,
                             names=my_rownames)

    ##########--------------
    # M.addConstrs((sum(G1_ub[w,t,k,tau]*D_lb[w,tau] for tau in range(T)) - \
    #           sum(G2_ub[w,t,k,tau]*D_ub[w,tau] for tau in range(T)) + \
    #           sum((F2_ub[w,t,k,tau]-F1_ub[w,t,k,tau])*E1[w,tau] for tau in range(T)) \
    #           >= a[k]*(q0[w]-tau_ub[t])+b[k]*alpha[t]-s0[w,t] \
    #           for w in range(W) for k in range(K) for t in range(T)), \
    #     name='cb1')
    rows = []
    my_rownames = ['cb1_%s_%s_%s' % (w, k, t) for w in range(W) for k in range(K) for t in range(T)]
    for w in range(W):
        for k in range(K):
            for t in range(T):
                row = [['G1_ub_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)] +
                       ['G2_ub_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)] +
                       ['F2_ub_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)] +
                       ['F1_ub_%s_%s_%s_%s' % (w, t, k, tau) for tau in range(T)]
                       + ['q0_%s' % (w)] + ['alpha_%s' % (t)] + ['s0_%s_%s' % (w, t)],

                       [float(D_lb[w, tau]) for tau in range(T)] + [float(-D_ub[w, tau]) for tau in range(T)]
                       + [float(E1[w, tau]) for tau in range(T)] + [float(-E1[w, tau]) for tau in range(T)]
                       + [-a[k]] + [-b[k]] + [1]]
                rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='G' * W * K * T,
                             rhs=[a[k] * (-float(tau_ub[t])) for w in range(W) for k in range(K) for t in range(T)],
                             names=my_rownames)

    # M.addConstrs((G1_ub[w,t,k,tau]- G2_ub[w,t,k,tau]-F1_ub[w,t,k,tau]+F2_ub[w,t,k,tau] \
    #               == (SS[w,t,tau] + a[k]*sum(unit_vector(T,m)[tau] for m in range(t+1))) \
    #               for w in range(W) for t in range(T) for k in range(K) for tau in range(T)), \
    #     name='cc1')
    rows = []
    my_rownames = ['cc1_%s_%s_%s_%s' % (w, t, k, tau) for w in range(W) for t in range(T) for k in range(K) for tau in
                   range(T)]
    for w in range(W):
        for t in range(T):
            for k in range(K):
                for tau in range(T):
                    row = [['G1_ub_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['G2_ub_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['F1_ub_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['F2_ub_%s_%s_%s_%s' % (w, t, k, tau)]
                           + ['SS_%s_%s_%s' % (w, t, tau)],

                           [1] + [-1]
                           + [-1] + [1]
                           + [-1]]
                    rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * T * K * T,
                             rhs=[a[k] * sum(unit_vector(T, m)[tau] for m in range(t + 1)) for w in range(W) for t in
                                  range(T) for k in range(K) for tau in range(T)],
                             names=my_rownames)

    # M.addConstrs((F1_ub[w,t,k,tau] + F2_ub[w,t,k,tau]  \
    #               == TT[w,t,tau]  \
    #               for w in range(W) for t in range(T) for k in range(K) for tau in range(T)),\
    #     name='cd1')
    rows = []
    my_rownames = ['cd1_%s_%s_%s_%s' % (w, t, k, tau) for w in range(W) for t in range(T) for k in range(K) for tau in
                   range(T)]
    for w in range(W):
        for t in range(T):
            for k in range(K):
                for tau in range(T):
                    row = [['F1_ub_%s_%s_%s_%s' % (w, t, k, tau)] +
                           ['F2_ub_%s_%s_%s_%s' % (w, t, k, tau)]
                           + ['TT_%s_%s_%s' % (w, t, tau)],

                           [1] + [1]
                           + [-1]]
                    rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * T * K * T,
                             rhs=[0] * W * T * K * T,
                             names=my_rownames)

    # M.optimize()
    M.write('1.lp')
    M.read('1.lp')
    M.solve()
    print(M.solution.get_objective_value())
    # print('quantity is')
    # print(q0[0].x)

    # return M.status
    return M.solution.get_status()


def valid_inequality_calculator(N, T, W, P, K, a, b, tau_lb, tau_ub, D_lb, D_ub, E1, E2):
    valid_inequality_parameter = np.ones(N, dtype=int) * T
    for n in range(N):
        for t in range(2, T + 1):
            status = single_node_calculator(t, W, P, K, a, b, tau_lb[n, :t], tau_ub[n, :t], D_lb[:, n, :t],
                                            D_ub[:, n, :t], E1[:, n, :t], E2[:, n, :t])
            # print(n,t)
            if status == 1:
                pass
            else:
                valid_inequality_parameter[n] = t - 1
                break

    return valid_inequality_parameter


def Scenario_ELDR_calculate(W, N, T, R, P, H, K, a, b, K_l, o, p, tau_lb, tau_ub, D_lb, D_ub, E1, E2, e1, e2, MM,
                            N_test, D_test, logTo_flag, valid_inequality_parameter, valid_flag, benders_flag):
    start = time.perf_counter()

    # M = gp.Model()
    # M.Params.LogToConsole=logTo_flag


    M = cplex.Cplex()
    M.objective.set_sense(M.objective.sense.minimize)

    # alpha = M.addVars(N,T,lb = 0.0001,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="alpha")
    M.variables.add(obj=[1] * N * T,
                    lb=[0.0001] * N * T,
                    ub=[cplex.infinity] * N * T,
                    types='C' * N * T,
                    names=['alpha_%s_%s' % (i, j) for i in range(N) for j in range(T)])

    # q0 = M.addVars(W,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="q0")
    M.variables.add(obj=[0] * W * N * T,
                    lb=[-cplex.infinity] * W * N * T,
                    ub=[cplex.infinity] * W * N * T,
                    types='C' * W * N * T,
                    names=['q0_%s_%s_%s' % (w, i, j) for w in range(W) for i in range(N) for j in range(T)])

    # Q0 = M.addVars(W,N,T,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="Q0")
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[-cplex.infinity] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['Q0_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T) for k
                           in range(N) for l in
                           range(T)])

    # x0 = M.addVars(W,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="x0")
    M.variables.add(obj=[0] * W * N * T,
                    lb=[-cplex.infinity] * W * N * T,
                    ub=[cplex.infinity] * W * N * T,
                    types='C' * W * N * T,
                    names=['x0_%s_%s_%s' % (w, i, j) for w in range(W) for i in range(N) for j in range(T)])

    # X0 = M.addVars(W,N,T,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="X0")
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[-cplex.infinity] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['X0_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # y = M.addVars(T,R,lb = 0, ub = 1,vtype=GRB.INTEGER,name="y")
    M.variables.add(obj=[0] * R * T,
                    lb=[0] * R * T,
                    ub=[1] * R * T,
                    types='B' * R * T,
                    names=['y_%s_%s' % (t, r) for t in range(T) for r in range(R)])

    # s0 = M.addVars(W,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="s0")
    M.variables.add(obj=[0] * W * N * T,
                    lb=[-cplex.infinity] * W * N * T,
                    ub=[cplex.infinity] * W * N * T,
                    types='C' * W * N * T,
                    names=['s0_%s_%s_%s' % (w, i, j) for w in range(W) for i in range(N) for j in range(T)])

    # SS = M.addVars(W,N,T,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="S")#为了和前面的S区分，命名为SS
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[-cplex.infinity] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['SS_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # r1 = M.addVars(W,N,T,T,vtype=GRB.CONTINUOUS,name="r")
    M.variables.add(obj=[0] * W * N * T * T,
                    lb=[0] * W * N * T * T,
                    ub=[cplex.infinity] * W * N * T * T,
                    types='C' * W * N * T * T,
                    names=['r1_%s_%s_%s_%s' % (w, i, j, k) for w in range(W) for i in range(N) for j in range(T) for k in
                           range(T)])

    # r2 = M.addVars(W,N,T,N,vtype=GRB.CONTINUOUS,name="r")
    M.variables.add(obj=[0] * W * N * T * N,
                    lb=[0] * W * N * T * N,
                    ub=[cplex.infinity] * W * N * T * N,
                    types='C' * W * N * T * N,
                    names=['r2_%s_%s_%s_%s' % (w, i, j, k) for w in range(W) for i in range(N) for j in range(T) for k
                           in range(N)])

    # TT = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="TT")#为了和前面的T区分，命名为TT
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[0] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['TT_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # G1_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G1_ub")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['G1_ub_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # G2_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G2_ub")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['G2_ub_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # F1_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F1_ub")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['F1_ub_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # F2_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F2_ub")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['F2_ub_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # w1_ub = M.addVars(W,N,K,T,K_l,T,vtype=GRB.CONTINUOUS,name="w1_ub")
    M.variables.add(obj=[0] * W * N * K * T * K_l * T,
                    lb=[0] * W * N * K * T * K_l * T,
                    ub=[cplex.infinity] * W * N * K * T * K_l * T,
                    types='C' * W * N * K * T * K_l * T,
                    names=['w1_ub_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(K) for k in range(T)
                           for
                           m in range(K_l) for l in range(T)])

    # w2_ub = M.addVars(W,N,K,N,K_l,T,vtype=GRB.CONTINUOUS,name="w2_ub")
    M.variables.add(obj=[0] * W * N * K * N * K_l * T,
                    lb=[0] * W * N * K * N * K_l * T,
                    ub=[cplex.infinity] * W * N * K * N * K_l * T,
                    types='C' * W * N * K * N * K_l * T,
                    names=['w2_ub_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(K) for k in range(N)
                           for
                           m in range(K_l) for l in range(T)])

    # G1_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G1_lb")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['G1_lb_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # G2_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G2_lb")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['G2_lb_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # F1_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F1_lb")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['F1_lb_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # F2_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F2_lb")
    M.variables.add(obj=[0] * W * N * T * K * N * T,
                    lb=[0] * W * N * T * K * N * T,
                    ub=[cplex.infinity] * W * N * T * K * N * T,
                    types='C' * W * N * T * N * K * T,
                    names=['F2_lb_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(T) for k in range(K)
                           for
                           m in range(N) for l in range(T)])

    # w1_lb = M.addVars(W,N,K,T,K_l,T,vtype=GRB.CONTINUOUS,name="w1_lb")
    M.variables.add(obj=[0] * W * N * K * T * K_l * T,
                    lb=[0] * W * N * K * T * K_l * T,
                    ub=[cplex.infinity] * W * N * K * T * K_l * T,
                    types='C' * W * N * K * T * K_l * T,
                    names=['w1_lb_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(K) for k in range(T)
                           for
                           m in range(K_l) for l in range(T)])

    # w2_lb = M.addVars(W,N,K,N,K_l,T,vtype=GRB.CONTINUOUS,name="w2_lb")
    M.variables.add(obj=[0] * W * N * K * N * K_l * T,
                    lb=[0] * W * N * K * N * K_l * T,
                    ub=[cplex.infinity] * W * N * K * N * K_l * T,
                    types='C' * W * N * K * N * K_l * T,
                    names=['w2_lb_%s_%s_%s_%s_%s_%s' % (w, i, j, k, m, l) for w in range(W) for i in range(N) for j in
                           range(K) for k in range(N)
                           for
                           m in range(K_l) for l in range(T)])

    # L_lb = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="L_lb")
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[0] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['L_lb_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # H_lb = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="H_lb")
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[0] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['H_lb_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # L_ub = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="L_ub")
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[0] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['L_ub_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # H_ub = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="H_ub")
    M.variables.add(obj=[0] * W * N * T * N * T,
                    lb=[0] * W * N * T * N * T,
                    ub=[cplex.infinity] * W * N * T * N * T,
                    types='C' * W * N * T * N * T,
                    names=['H_ub_%s_%s_%s_%s_%s' % (w, i, j, k, l) for w in range(W) for i in range(N) for j in range(T)
                           for k in range(N) for l in
                           range(T)])

    # M.setObjective(gp.quicksum(alpha[n,t] for n in range(N) for t in range(T)),GRB.MINIMIZE)

    # M.addConstrs((sum(P[w]*(s0[w,n,t] + \
    #               sum(SS[w,n,t,i,tau]*E1[w,i,tau] for i in range(N) for tau in range(T)) + \
    #               sum(TT[w,n,t,i,tau]*E2[w,i,tau] for i in range(N) for tau in range(T)) + \
    #               sum(r1[w,n,t,tau]*e1[w,tau] for tau in range(T))+ \
    #               sum(r2[w,n,t,i]*e2[w,i] for i in range(N))) for w in range(W)) \
    #               <= 0 \
    #               for n in range(N) for t in range(T)),\
    #     name='ca0')
    rows = []
    my_rownames = ['ca0_%s_%s' % (n, t) for n in range(N) for t in range(T)]
    for n in range(N):
        for t in range(T):
            row = [['s0_%s_%s_%s' % (w, n, t) for w in range(W)] +
                   ['SS_%s_%s_%s_%s_%s' % (w, n, t, i, tau) for i in range(N) for tau in range(T) for w in range(W)] +
                   ['TT_%s_%s_%s_%s_%s' % (w, n, t, i, tau) for i in range(N) for tau in range(T) for w in range(W)] +
                   ['r1_%s_%s_%s_%s' % (w, n, t, tau) for tau in range(T) for w in range(W)] +
                   ['r2_%s_%s_%s_%s' % (w, n, t, i) for i in range(N) for w in range(W)],
                   [float(P[w]) for w in range(W)] + [float(P[w] * E1[w, i, tau]) for i in range(N) for tau in range(T) for
                                                      w in range(W)] +
                   [float(P[w] * E2[w, i, tau]) for i in range(N) for tau in range(T) for w in range(W)] +
                   [float(P[w] * e1[w, tau]) for tau in range(T) for w in range(W)] +
                   [float(P[w] * e2[w, i]) for i in range(N) for w in range(W)]]
            rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='L' * N * T,
                             rhs=[0] * N * T,
                             names=my_rownames)

    ##########--------------
    # M.addConstrs((sum(G1_lb[w,n,t,k,i,tau]*D_lb[w,i,tau] for i in range(N) for tau in range(T)) - \
    #               sum(G2_lb[w,n,t,k,i,tau]*D_ub[w,i,tau] for i in range(N) for tau in range(T)) + \
    #               sum((F2_lb[w,n,t,k,i,tau]-F1_lb[w,n,t,k,i,tau])*E1[w,i,tau] for i in range(N) for tau in range(T)) + \
    #               sum(sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for i in range(N)))*w1_lb[w,n,k,tau,kk,t] for tau in range(T)) + \
    #               sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for tau in range(T)))*w2_lb[w,n,k,i,kk,t] for i in range(N)) for kk in range(K_l)) \
    #               >= a[k]*(tau_lb[n,t]-x0[w,n,t])+b[k]*alpha[n,t]-s0[w,n,t] \
    #               for w in range(W) for k in range(K) for n in range(N) for t in range(T)), \
    #     name='cb0')
    rows = []
    my_rownames = ['cb0_%s_%s_%s_%s' % (w, k, n, t) for w in range(W) for k in range(K) for n in range(N) for t in range(T)]
    for w in range(W):
        for k in range(K):
            for n in range(N):
                for t in range(T):
                    row = [['G1_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['G2_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['F2_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['F1_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['w1_lb_%s_%s_%s_%s_%s_%s' % (w, n, k, tau, kk, t) for tau in range(T) for kk in range(K_l)] +
                           ['w2_lb_%s_%s_%s_%s_%s_%s' % (w, n, k, i, kk, t) for i in range(N) for kk in range(K_l)] +
                           ['x0_%s_%s_%s' % (w, n, t)] + ['alpha_%s_%s' % (n, t)] + ['s0_%s_%s_%s' % (w, n, t)],

                           [float(D_lb[w, i, tau]) for i in range(N) for tau in range(T)] + [float(-D_ub[w, i, tau]) for i
                                                                                             in range(N) for tau in
                                                                                             range(T)]
                           + [float(E1[w, i, tau]) for i in range(N) for tau in range(T)] + [float(-E1[w, i, tau]) for i in
                                                                                             range(N) for tau in range(T)]
                           + [float((p[kk] - o[kk] * sum(E1[w, i, tau] / E2[w, i, tau] for i in range(N)))) for tau in
                              range(T) for kk in range(K_l)]
                           + [float((p[kk] - o[kk] * sum(E1[w, i, tau] / E2[w, i, tau] for tau in range(T)))) for i in
                              range(N) for kk in range(K_l)]
                           + [a[k]] + [-b[k]] + [1]]
                    rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='G' * W * K * N * T,
                             rhs=[a[k] * (float(tau_lb[n, t])) for w in range(W) for k in range(K) for n in range(N) for t
                                  in range(T)],
                             names=my_rownames)

    # M.addConstrs((G1_lb[w,n,t,k,i,tau]- G2_lb[w,n,t,k,i,tau]-F1_lb[w,n,t,k,i,tau]+F2_lb[w,n,t,k,i,tau] - \
    #               sum(o[kk]/E2[w,i,tau]*(w1_lb[w,n,k,tau,kk,t]+w2_lb[w,n,k,i,kk,t]) for kk in range(K_l)) \
    #               == (SS[w,n,t,i,tau] + a[k]*X0[w,n,t,i,tau]) \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)), \
    #     name='cc0')
    rows = []
    my_rownames = ['cc0_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for w in range(W) for n in range(N) for t in range(T) for
                   k in range(K) for i in range(N) for tau in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for i in range(N):
                        for tau in range(T):
                            row = [['G1_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['G2_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['F1_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['F2_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['w1_lb_%s_%s_%s_%s_%s_%s' % (w, n, k, tau, kk, t) for kk in range(K_l)] +
                                   ['w2_lb_%s_%s_%s_%s_%s_%s' % (w, n, k, i, kk, t) for kk in range(K_l)]
                                   + ['SS_%s_%s_%s_%s_%s' % (w, n, t, i, tau)] +
                                   ['X0_%s_%s_%s_%s_%s' % (w, n, t, i, tau)],

                                   [1] + [-1]
                                   + [-1] + [1]
                                   + [-float(o[kk] / E2[w, i, tau]) for kk in range(K_l)]
                                   + [-float(o[kk] / E2[w, i, tau]) for kk in range(K_l)]
                                   + [-1] + [-a[k]]]
                            rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * N * T,
                             rhs=[0] * W * N * T * K * N * T,
                             names=my_rownames)

    # M.addConstrs((F1_lb[w,n,t,k,i,tau] + F2_lb[w,n,t,k,i,tau] \
    #               == TT[w,n,t,i,tau] \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)),\
    #     name='cd0')
    rows = []
    my_rownames = ['cd0_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for w in range(W) for n in range(N) for t in range(T) for
                   k in range(K) for i in range(N) for tau in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for i in range(N):
                        for tau in range(T):
                            row = [['F1_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['F2_lb_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)]
                                   + ['TT_%s_%s_%s_%s_%s' % (w, n, t, i, tau)],

                                   [1] + [1]
                                   + [-1]]
                            rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * N * T,
                             rhs=[0] * W * N * T * K * N * T,
                             names=my_rownames)

    # M.addConstrs((sum(w1_lb[w,n,k,tau,kk,t] for kk in range(K_l))\
    #               == r1[w,n,t,tau]  \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for tau in range(T)),\
    #     name='cd2')
    rows = []
    my_rownames = ['ce0_%s_%s_%s_%s_%s' % (w, n, t, k, tau) for w in range(W) for n in range(N) for t in range(T)
                   for k in range(K) for tau in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for tau in range(T):
                        row = [['w1_lb_%s_%s_%s_%s_%s_%s' % (w, n, k, tau, kk, t) for kk in range(K_l)]
                               + ['r1_%s_%s_%s_%s' % (w, n, t, tau)],

                               [1 for i in range(K_l)]
                               + [-1]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * T,
                             rhs=[0] * W * N * T * K * T,
                             names=my_rownames)

    # M.addConstrs((sum(w2_lb[w,n,k,i,kk,t] for kk in range(K_l))\
    #               == r2[w,n,t,i]  \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N)),\
    #     name='cd2')
    rows = []
    my_rownames = ['cf0_%s_%s_%s_%s_%s' % (w, n, t, k, i) for w in range(W) for n in range(N) for t in range(T)
                   for k in range(K) for i in range(N) ]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for i in range(N):
                        row = [['w2_lb_%s_%s_%s_%s_%s_%s' % (w, n, k, i, kk, t) for kk in range(K_l)]
                               + ['r2_%s_%s_%s_%s' % (w, n, t, i)],

                               [1 for i in range(K_l)]
                               + [-1]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * N,
                             rhs=[0] * W * N * T * K * N,
                             names=my_rownames)

    ##########--------------
    # M.addConstrs((sum(G1_ub[w,n,t,k,i,tau]*D_lb[w,i,tau] for i in range(N) for tau in range(T)) - \
    #           sum(G2_ub[w,n,t,k,i,tau]*D_ub[w,i,tau] for i in range(N) for tau in range(T)) + \
    #           sum((F2_ub[w,n,t,k,i,tau]-F1_ub[w,n,t,k,i,tau])*E1[w,i,tau] for i in range(N) for tau in range(T)) + \
    #           sum(sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for i in range(N)))*w1_ub[w,n,k,tau,kk,t] for tau in range(T)) + \
    #           sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for tau in range(T)))*w2_ub[w,n,k,i,kk,t] for i in range(N)) for kk in range(K_l)) \
    #           >= a[k]*(x0[w,n,t]-tau_ub[n,t])+b[k]*alpha[n,t]-s0[w,n,t] \
    #           for w in range(W) for k in range(K) for n in range(N) for t in range(T)), \
    #     name='cb1')
    rows = []
    my_rownames = ['cb1_%s_%s_%s_%s' % (w, k, n, t) for w in range(W) for k in range(K) for n in range(N) for t in range(T)]
    for w in range(W):
        for k in range(K):
            for n in range(N):
                for t in range(T):
                    row = [['G1_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['G2_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['F2_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['F1_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for i in range(N) for tau in range(T)] +
                           ['w1_ub_%s_%s_%s_%s_%s_%s' % (w, n, k, tau, kk, t) for tau in range(T) for kk in range(K_l)] +
                           ['w2_ub_%s_%s_%s_%s_%s_%s' % (w, n, k, i, kk, t) for i in range(N) for kk in range(K_l)] +
                           ['x0_%s_%s_%s' % (w, n, t)] + ['alpha_%s_%s' % (n, t)] + ['s0_%s_%s_%s' % (w, n, t)],

                           [float(D_lb[w, i, tau]) for i in range(N) for tau in range(T)] + [float(-D_ub[w, i, tau]) for i
                                                                                             in range(N) for tau in
                                                                                             range(T)]
                           + [float(E1[w, i, tau]) for i in range(N) for tau in range(T)] + [float(-E1[w, i, tau]) for i in
                                                                                             range(N) for tau in range(T)]
                           + [float((p[kk] - o[kk] * sum(E1[w, i, tau] / E2[w, i, tau] for i in range(N)))) for tau in
                              range(T) for kk in range(K_l)]
                           + [float((p[kk] - o[kk] * sum(E1[w, i, tau] / E2[w, i, tau] for tau in range(T)))) for i in
                              range(N) for kk in range(K_l)]
                           + [-a[k]] + [-b[k]] + [1]]
                    rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='G' * W * K * N * T,
                             rhs=[a[k] * (float(-tau_ub[n, t])) for w in range(W) for k in range(K) for n in range(N) for t
                                  in range(T)],
                             names=my_rownames)

    # M.addConstrs((G1_ub[w,n,t,k,i,tau]- G2_ub[w,n,t,k,i,tau]-F1_ub[w,n,t,k,i,tau]+F2_ub[w,n,t,k,i,tau] - \
    #               sum(o[kk]/E2[w,i,tau]*(w1_ub[w,n,k,tau,kk,t]+w2_ub[w,n,k,i,kk,t]) for kk in range(K_l)) \
    #               == (SS[w,n,t,i,tau] - a[k]*X0[w,n,t,i,tau]) \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)), \
    #     name='cc1')
    rows = []
    my_rownames = ['cc1_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for w in range(W) for n in range(N) for t in range(T)
                   for k in range(K) for i in range(N) for tau in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for i in range(N):
                        for tau in range(T):
                            row = [['G1_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['G2_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['F1_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['F2_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['w1_ub_%s_%s_%s_%s_%s_%s' % (w, n, k, tau, kk, t) for kk in range(K_l)] +
                                   ['w2_ub_%s_%s_%s_%s_%s_%s' % (w, n, k, i, kk, t) for kk in range(K_l)]
                                   + ['SS_%s_%s_%s_%s_%s' % (w, n, t, i, tau)] +
                                   ['X0_%s_%s_%s_%s_%s' % (w, n, t, i, tau)],

                                   [1] + [-1]
                                   + [-1] + [1]
                                   + [-float(o[kk] / E2[w, i, tau]) for kk in range(K_l)]
                                   + [-float(o[kk] / E2[w, i, tau]) for kk in range(K_l)]
                                   + [-1] + [a[k]]]
                            rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * N * T,
                             rhs=[0] * W * N * T * K * N * T,
                             names=my_rownames)

    # M.addConstrs((F1_ub[w,n,t,k,i,tau] + F2_ub[w,n,t,k,i,tau] \
    #               == TT[w,n,t,i,tau] \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)),\
    #     name='cd1')
    rows = []
    my_rownames = ['cd1_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau) for w in range(W) for n in range(N) for t in range(T)
                   for k in range(K) for i in range(N) for tau in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for i in range(N):
                        for tau in range(T):
                            row = [['F1_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)] +
                                   ['F2_ub_%s_%s_%s_%s_%s_%s' % (w, n, t, k, i, tau)]
                                   + ['TT_%s_%s_%s_%s_%s' % (w, n, t, i, tau)],

                                   [1] + [1]
                                   + [-1]]
                            rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * N * T,
                             rhs=[0] * W * N * T * K * N * T,
                             names=my_rownames)

    # M.addConstrs((sum(w1_ub[w,n,k,tau,kk,t] for kk in range(K_l))\
    #               == r1[w,n,t,tau]  \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for tau in range(T)),\
    #     name='cd2')
    rows = []
    my_rownames = ['ce1_%s_%s_%s_%s_%s' % (w, n, t, k, tau) for w in range(W) for n in range(N) for t in range(T)
                   for k in range(K) for tau in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for tau in range(T):
                        row = [['w1_ub_%s_%s_%s_%s_%s_%s' % (w, n, k, tau, kk, t) for kk in range(K_l)]
                               + ['r1_%s_%s_%s_%s' % (w, n, t, tau)],

                               [1 for k in range(K_l)]
                               + [-1]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * T,
                             rhs=[0] * W * N * T * K * T,
                             names=my_rownames)

    # M.addConstrs((sum(w2_ub[w,n,k,i,kk,t] for kk in range(K_l))\
    #               == r2[w,n,t,i]  \
    #               for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N)),\
    #     name='cd2')
    rows = []
    my_rownames = ['cf1_%s_%s_%s_%s_%s' % (w, n, t, k, i) for w in range(W) for n in range(N) for t in range(T)
                   for k in range(K) for i in range(N)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for k in range(K):
                    for i in range(N):
                        row = [['w2_ub_%s_%s_%s_%s_%s_%s' % (w, n, k, i, kk, t) for kk in range(K_l)]
                               + ['r2_%s_%s_%s_%s' % (w, n, t, i)],

                               [1] * K_l
                               + [-1]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * K * N,
                             rhs=[0] * W * N * T * K * N,
                             names=my_rownames)

    # ----------------------------------------------------------

    # M.addConstrs((-q0[w,n,t]+sum(D_ub[w,i,j]*L_lb[w,n,t,i,j] for i in range(N) for j in range(T))- \
    #               sum(D_lb[w,i,j]*H_lb[w,n,t,i,j] for i in range(N) for j in range(T)) \
    #               <=0 for w in range(W) for n in range(N) for t in range(T)),\
    #     name="cb0")
    rows = []
    my_rownames = ['cg0_%s_%s_%s' % (w, n, t) for w in range(W) for n in range(N) for t in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                row = [['q0_%s_%s_%s' % (w, n, t)] + ['L_lb_%s_%s_%s_%s_%s' % (w, n, t, i, j) for i in range(N) for j in
                                                      range(T)] + [
                           'H_lb_%s_%s_%s_%s_%s' % (w, n, t, i, j) for i in range(N) for j in range(T)],
                       [-1] + [float(D_ub[w, i, j]) for i in range(N) for j in range(T)] + [-float(D_lb[w, i, j]) for i in
                                                                                            range(N)
                                                                                            for j in range(T)]]
                rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='L' * W * N * T,
                             rhs=[0] * W * N * T,
                             names=my_rownames)

    # M.addConstrs((Q0[w,n,t,i,j]+L_lb[w,n,t,i,j]-H_lb[w,n,t,i,j]\
    #               ==0 for w in range(W) for n in range(N) for t in range(T) for i in range(N) for j in range(T) ),\
    #     name="cb1")
    rows = []
    my_rownames = ['ch0_%s_%s_%s_%s_%s' % (w, n, t, i, j) for w in range(W) for n in range(N) for t in range(T) for i in
                   range(N) for j in
                   range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for i in range(N):
                    for j in range(T):
                        row = [['Q0_%s_%s_%s_%s_%s' % (w, n, t, i, j)] + ['L_lb_%s_%s_%s_%s_%s' % (w, n, t, i, j)] + [
                            'H_lb_%s_%s_%s_%s_%s' % (w, n, t, i, j)],
                               [1] + [1] + [-1]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * N * T,
                             rhs=[0] * W * N * T * N * T,
                             names=my_rownames)

    # M.addConstrs((q0[w,n,t]+sum(D_ub[w,i,j]*L_ub[w,n,t,i,j] for i in range(N) for j in range(T))- \
    #               sum(D_lb[w,i,j]*H_ub[w,n,t,i,j] for i in range(N) for j in range(T)) \
    #               <=MM*sum(H[r][n]*y[t,r] for r in range(R)) for w in range(W) for n in range(N) for t in range(1,T)),\
    #     name="cb3")
    rows = []
    my_rownames = ['ci0_%s_%s_%s' % (w, n, t) for w in range(W) for n in range(N) for t in range(1, T)]
    for w in range(W):
        for n in range(N):
            for t in range(1, T):
                row = [['q0_%s_%s_%s' % (w, n, t)] + ['L_ub_%s_%s_%s_%s_%s' % (w, n, t, i, j) for i in range(N) for j in
                                                      range(T)] + [
                           'H_ub_%s_%s_%s_%s_%s' % (w, n, t, i, j) for i in range(N) for j in range(T)]
                       + ['y_%s_%s' % (t, r) for r in range(R)],
                       [1] + [float(D_ub[w, i, j]) for i in range(N) for j in range(T)] + [-float(D_lb[w, i, j]) for i in
                                                                                           range(N)
                                                                                           for
                                                                                           j in range(T)]
                       + [-MM * H[r][n] for r in range(R)]]
                rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='L' * W * N * (T - 1),
                             rhs=[0] * W * N * (T - 1),
                             names=my_rownames)

    # M.addConstrs((Q0[w,n,t,i,j]-L_ub[w,n,t,i,j]+H_ub[w,n,t,i,j]\
    #               ==0 for w in range(W) for n in range(N) for t in range(T) for i in range(N) for j in range(T) ),\
    #     name="cb4")
    rows = []
    my_rownames = ['cj0_%s_%s_%s_%s_%s' % (w, n, t, i, j) for w in range(W) for n in range(N) for t in range(T) for i in
                   range(N) for j in
                   range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for i in range(N):
                    for j in range(T):
                        row = [['Q0_%s_%s_%s_%s_%s' % (w, n, t, i, j)] + ['L_ub_%s_%s_%s_%s_%s' % (w, n, t, i, j)] + [
                            'H_ub_%s_%s_%s_%s_%s' % (w, n, t, i, j)],
                               [1] + [-1] + [1]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * N * T,
                             rhs=[0] * W * N * T * N * T,
                             names=my_rownames)

    # M.addConstrs((x0[w,n,t]==sum(q0[w,n,m] for m in range(t+1))\
    #               for w in range(W) for n in range(N) for t in range(T)),name='ca0')
    rows = []
    my_rownames = ['ck0_%s_%s_%s' % (w, n, t) for w in range(W) for n in range(N) for t in range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                row = [['x0_%s_%s_%s' % (w, n, t)] +
                       ['q0_%s_%s_%s' % (w, n, m) for m in range(t + 1)],
                       [1] + [-1 for m in range(t + 1)]]
                rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T,
                             rhs=[0] * W * N * T,
                             names=my_rownames)

    # M.addConstrs((X0[w,n,t,i,tau]==sum(Q0[w,n,m,i,tau] - e(N,T,n,m)[i,tau] for m in range(t+1))\
    #               for w in range(W) for n in range(N) for t in range(T) for i in range(N) for tau in range(T)),name='ca1')
    rows = []
    my_rownames = ['cl0_%s_%s_%s_%s_%s' % (w, n, t, i, j) for w in range(W) for n in range(N) for t in range(T) for i in
                   range(N) for j in
                   range(T)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for i in range(N):
                    for j in range(T):
                        row = [['X0_%s_%s_%s_%s_%s' % (w, n, t, i, j)] + ['Q0_%s_%s_%s_%s_%s' % (w, n, m, i, j) for m in
                                                                          range(t + 1)],
                               [1] + [-1 for m in range(t + 1)]]
                        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * W * N * T * N * T,
                             rhs=[sum(- e(N, T, n, m)[i, j] for m in range(t + 1)) for w in range(W) for n in range(N) for t
                                  in
                                  range(T) for i in range(N) for j in range(T)],
                             names=my_rownames)

    # M.addConstrs((sum(y[t,r] for r in range(R)) == 1 for t in range(1,T)),name='ca2')
    rows = []
    my_rownames = ['cm0_%s' % (t) for t in range(1, T)]
    for t in range(1, T):
        row = [['y_%s_%s' % (t, r) for r in range(R)],
               [1 for r in range(R)]]
        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * (T - 1),
                             rhs=[1] * (T - 1),
                             names=my_rownames)

    # M.addConstrs((y[0,r] == 0 for r in range(R)), name="cy1")
    rows = []
    my_rownames = ['cn0_%s' % (r) for r in range(R)]
    for r in range(R):
        row = [['y_0_%s' % (r)],
               [1]]
        rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * R,
                             rhs=[0] * R,
                             names=my_rownames)

    # M.addConstrs((Q0[w,n,t,i,l] == 0 \
    #               for w in range(W) for n in range(N) for t in range(T) \
    #               for l in range(T) if l>=t for i in range(N)),\
    #         name="cg2")
    rows = []
    my_rownames = ['co0_%s_%s_%s_%s_%s' % (w, n, t, i, l) for w in range(W) for n in range(N) for t in range(T) for l in
                   range(T) if l >= t for
                   i in
                   range(N)]
    for w in range(W):
        for n in range(N):
            for t in range(T):
                for l in range(T):
                    if l >= t:
                        for i in range(N):
                            row = [['Q0_%s_%s_%s_%s_%s' % (w, n, t, i, l)],
                                   [1]]
                            rows.append(row)
    M.linear_constraints.add(lin_expr=rows,
                             senses='E' * len(my_rownames),
                             rhs=[0] * len(my_rownames),
                             names=my_rownames)

    if valid_flag:
        for n in range(N):
            # M.addConstrs((sum(H[r][n]*y[tau,r] for r in range(R) for tau in range(t,t+valid_inequality_parameter[n]))
            #           >= 1 for t in range(1,T-valid_inequality_parameter[n]+1)),
            # name="h0")
            rows = []
            my_rownames = ['h0_%s' % (t) for t in range(1, T - valid_inequality_parameter[n] + 1)]
            for t in range(1, T - valid_inequality_parameter[n] + 1):
                row = [['y_%s_%s' % (tau, r) for r in range(R) for tau in range(t, t + valid_inequality_parameter[n])],
                       [float(H[r][n]) for r in range(R) for tau in range(t, t + valid_inequality_parameter[n])]]
                rows.append(row)
            M.linear_constraints.add(lin_expr=rows,
                                     senses='G' * len(my_rownames),
                                     rhs=[1] * len(my_rownames),
                                     names=my_rownames)

    # M.write("asdf.lp")
    # M.optimize()
    
    if benders_flag:
        M.parameters.benders.strategy.set(M.parameters.benders.strategy.values.full)
        M.write_benders_annotation('benders.ann')
    
    M.solve()

    obje = M.solution.get_objective_value()

    solve_time = time.perf_counter() - start

    y_t = np.zeros((T, R))
    for t in range(T):
        for r in range(R):
            # y_t[t, r] = y[t, r].x
            y_t[t,r]=M.solution.get_values('y_%s_%s'%(t,r))

    alpha_t = np.zeros((N, T))
    for n in range(N):
        for t in range(T):
            # alpha_t[n, t] = alpha[n, t].x
            alpha_t[n,t]=M.solution.get_values('alpha_%s_%s'%(n,t))

    if W == 1:
        q_t = {w: np.zeros((N, T, N_test[w])) for w in range(len(D_test))}
        x_t = {w: np.zeros((N, T, N_test[w])) for w in range(len(D_test))}
        v_t = {w: np.zeros((N, T, N_test[w])) for w in range(len(D_test))}
        p_t = {w: np.zeros((N, T, N_test[w])) for w in range(len(D_test))}
        for w in range(len(D_test)):
            for n in range(N):
                for t in range(T):
                    for i in range(N_test[w]):
                        # q_t[w][n, t, i] = q0[0, n, t].x + sum(
                        #     Q0[0, n, t, nn, tt].x * D_test[w][nn, tt, i] for nn in range(N) for tt in range(T))
                        # x_t[w][n, t, i] = x0[0, n, t].x + sum(
                        #     X0[0, n, t, nn, tt].x * D_test[w][nn, tt, i] for nn in range(N) for tt in range(T))
                        # v_t[w][n, t, i] = max(x_t[w][n, t, i] - tau_ub[n, t], tau_lb[n, t] - x_t[w][n, t, i], 0)
                        # p_t[w][n, t, i] = abs(v_t[w][n, t, i]) >= 0.0001
                        q_t[w][n, t, i] = M.solution.get_values('q0_%s_%s_%s'%(0, n, t))+ sum(
                            M.solution.get_values('Q0_%s_%s_%s_%s_%s'%(0, n, t, nn, tt)) * D_test[w][nn, tt, i] for nn in range(N) for tt in range(T))
                        x_t[w][n, t, i] = M.solution.get_values('x0_%s_%s_%s'%(0, n, t)) + sum(
                            M.solution.get_values('X0_%s_%s_%s_%s_%s' % (0, n, t, nn, tt)) * D_test[w][nn, tt, i] for nn in range(N) for tt in range(T))
                        v_t[w][n, t, i] = max(x_t[w][n, t, i] - tau_ub[n, t], tau_lb[n, t] - x_t[w][n, t, i], 0)
                        p_t[w][n, t, i] = abs(v_t[w][n, t, i]) >= 0.0001
    else:
        q_t = {w: np.zeros((N, T, N_test[w])) for w in range(W)}
        x_t = {w: np.zeros((N, T, N_test[w])) for w in range(W)}
        v_t = {w: np.zeros((N, T, N_test[w])) for w in range(W)}
        p_t = {w: np.zeros((N, T, N_test[w])) for w in range(W)}
        for w in range(W):
            for n in range(N):
                for t in range(T):
                    for i in range(N_test[w]):
                        q_t[w][n, t, i] = M.solution.get_values('q0_%s_%s_%s'%(w, n, t)) + sum(
                            M.solution.get_values('Q0_%s_%s_%s_%s_%s'%(w, n, t, nn, tt)) * D_test[w][nn, tt, i] for nn in range(N) for tt in range(T))
                        x_t[w][n, t, i] =  M.solution.get_values('x0_%s_%s_%s'%(w, n, t)) + sum(
                            M.solution.get_values('X0_%s_%s_%s_%s_%s'%(w, n, t, nn, tt)) * D_test[w][nn, tt, i] for nn in range(N) for tt in range(T))
                        v_t[w][n, t, i] = max(x_t[w][n, t, i] - tau_ub[n, t], tau_lb[n, t] - x_t[w][n, t, i], 0)
                        p_t[w][n, t, i] = abs(v_t[w][n, t, i]) >= 0.0001

    return obje, alpha_t, y_t, q_t, x_t, v_t, p_t, solve_time

# def single_Scenario_ELDR_calculate(nn,W,N,T,R,P,H,K,a,b,K_l,o,p,tau_lb,tau_ub,D_lb,D_ub,E1,E2,e1,e2,MM,N_test,D_test,logTo_flag,valid_inequality_parameter,valid_flag):


#     start = time.perf_counter()

#     M = gp.Model()
#     M.Params.LogToConsole=logTo_flag


#     alpha = M.addVars(N,T,lb = 0.0001,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="alpha")

#     q0 = M.addVars(W,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="q0")
#     Q0 = M.addVars(W,N,T,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="Q0")

#     x0 = M.addVars(W,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="x0")
#     X0 = M.addVars(W,N,T,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="X0")

#     y = M.addVars(T,R,lb = 0, ub = 1,vtype=GRB.INTEGER,name="y")

#     s0 = M.addVars(W,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="s0")
#     SS = M.addVars(W,N,T,N,T,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="S")#为了和前面的S区分，命名为SS
#     r1 = M.addVars(W,N,T,T,vtype=GRB.CONTINUOUS,name="r")
#     r2 = M.addVars(W,N,T,N,vtype=GRB.CONTINUOUS,name="r")
#     TT = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="TT")#为了和前面的T区分，命名为TT

#     G1_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G1_ub")
#     G2_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G2_ub")
#     F1_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F1_ub")
#     F2_ub = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F2_ub")
#     w1_ub = M.addVars(W,N,K,T,K_l,T,vtype=GRB.CONTINUOUS,name="w1_ub")
#     w2_ub = M.addVars(W,N,K,N,K_l,T,vtype=GRB.CONTINUOUS,name="w2_ub")

#     G1_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G1_lb")
#     G2_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="G2_lb")
#     F1_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F1_lb")
#     F2_lb = M.addVars(W,N,T,K,N,T,vtype=GRB.CONTINUOUS,name="F2_lb")
#     w1_lb = M.addVars(W,N,K,T,K_l,T,vtype=GRB.CONTINUOUS,name="w1_lb")
#     w2_lb = M.addVars(W,N,K,N,K_l,T,vtype=GRB.CONTINUOUS,name="w2_lb")


#     L_lb = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="L_lb")
#     H_lb = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="H_lb")
#     L_ub = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="L_ub")
#     H_ub = M.addVars(W,N,T,N,T,vtype=GRB.CONTINUOUS,name="H_ub")


#     M.setObjective(sum(alpha[nn,t] for t in range(T)),GRB.MINIMIZE)

#     M.addConstrs((sum(P[w]*(s0[w,n,t] + \
#                   sum(SS[w,n,t,i,tau]*E1[w,i,tau] for i in range(N) for tau in range(T)) + \
#                   sum(TT[w,n,t,i,tau]*E2[w,i,tau] for i in range(N) for tau in range(T)) + \
#                   sum(r1[w,n,t,tau]*e1[w,tau] for tau in range(T))+ \
#                   sum(r2[w,n,t,i]*e2[w,i] for i in range(N))) for w in range(W)) \
#                   <= 0 \
#                   for n in range(N) for t in range(T)),\
#         name='ca0')


#     ##########-------------- 
#     M.addConstrs((sum(G1_lb[w,n,t,k,i,tau]*D_lb[w,i,tau] for i in range(N) for tau in range(T)) - \
#                   sum(G2_lb[w,n,t,k,i,tau]*D_ub[w,i,tau] for i in range(N) for tau in range(T)) + \
#                   sum((F2_lb[w,n,t,k,i,tau]-F1_lb[w,n,t,k,i,tau])*E1[w,i,tau] for i in range(N) for tau in range(T)) + \
#                   sum(sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for i in range(N)))*w1_lb[w,n,k,tau,kk,t] for tau in range(T)) + \
#                   sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for tau in range(T)))*w2_lb[w,n,k,i,kk,t] for i in range(N)) for kk in range(K_l)) \
#                   >= a[k]*(tau_lb[n,t]-x0[w,n,t])+b[k]*alpha[n,t]-s0[w,n,t] \
#                   for w in range(W) for k in range(K) for n in range(N) for t in range(T)), \
#         name='cb0')

#     M.addConstrs((G1_lb[w,n,t,k,i,tau]- G2_lb[w,n,t,k,i,tau]-F1_lb[w,n,t,k,i,tau]+F2_lb[w,n,t,k,i,tau] - \
#                   sum(o[kk]/E2[w,i,tau]*(w1_lb[w,n,k,tau,kk,t]+w2_lb[w,n,k,i,kk,t]) for kk in range(K_l)) \
#                   == (SS[w,n,t,i,tau] + a[k]*X0[w,n,t,i,tau]) \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)), \
#         name='cc0')

#     M.addConstrs((F1_lb[w,n,t,k,i,tau] + F2_lb[w,n,t,k,i,tau] \
#                   == TT[w,n,t,i,tau] \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)),\
#         name='cd0')

#     M.addConstrs((sum(w1_lb[w,n,k,tau,kk,t] for kk in range(K_l))\
#                   == r1[w,n,t,tau]  \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for tau in range(T)),\
#         name='cd2')

#     M.addConstrs((sum(w2_lb[w,n,k,i,kk,t] for kk in range(K_l))\
#                   == r2[w,n,t,i]  \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N)),\
#         name='cd2')

#     ##########-------------- 
#     M.addConstrs((sum(G1_ub[w,n,t,k,i,tau]*D_lb[w,i,tau] for i in range(N) for tau in range(T)) - \
#               sum(G2_ub[w,n,t,k,i,tau]*D_ub[w,i,tau] for i in range(N) for tau in range(T)) + \
#               sum((F2_ub[w,n,t,k,i,tau]-F1_ub[w,n,t,k,i,tau])*E1[w,i,tau] for i in range(N) for tau in range(T)) + \
#               sum(sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for i in range(N)))*w1_ub[w,n,k,tau,kk,t] for tau in range(T)) + \
#               sum((p[kk]-o[kk]*sum(E1[w,i,tau]/E2[w,i,tau] for tau in range(T)))*w2_ub[w,n,k,i,kk,t] for i in range(N)) for kk in range(K_l)) \
#               >= a[k]*(x0[w,n,t]-tau_ub[n,t])+b[k]*alpha[n,t]-s0[w,n,t] \
#               for w in range(W) for k in range(K) for n in range(N) for t in range(T)), \
#         name='cb1')  

#     M.addConstrs((G1_ub[w,n,t,k,i,tau]- G2_ub[w,n,t,k,i,tau]-F1_ub[w,n,t,k,i,tau]+F2_ub[w,n,t,k,i,tau] - \
#                   sum(o[kk]/E2[w,i,tau]*(w1_ub[w,n,k,tau,kk,t]+w2_ub[w,n,k,i,kk,t]) for kk in range(K_l)) \
#                   == (SS[w,n,t,i,tau] - a[k]*X0[w,n,t,i,tau]) \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)), \
#         name='cc1')

#     M.addConstrs((F1_ub[w,n,t,k,i,tau] + F2_ub[w,n,t,k,i,tau] \
#                   == TT[w,n,t,i,tau] \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N) for tau in range(T)),\
#         name='cd1')

#     M.addConstrs((sum(w1_ub[w,n,k,tau,kk,t] for kk in range(K_l))\
#                   == r1[w,n,t,tau]  \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for tau in range(T)),\
#         name='cd2')

#     M.addConstrs((sum(w2_ub[w,n,k,i,kk,t] for kk in range(K_l))\
#                   == r2[w,n,t,i]  \
#                   for w in range(W) for n in range(N) for t in range(T) for k in range(K) for i in range(N)),\
#         name='cd2')

#     #----------------------------------------------------------

#     M.addConstrs((-q0[w,n,t]+sum(D_ub[w,i,j]*L_lb[w,n,t,i,j] for i in range(N) for j in range(T))- \
#                   sum(D_lb[w,i,j]*H_lb[w,n,t,i,j] for i in range(N) for j in range(T)) \
#                   <=0 for w in range(W) for n in range(N) for t in range(T)),\
#         name="cb0")

#     M.addConstrs((Q0[w,n,t,i,j]+L_lb[w,n,t,i,j]-H_lb[w,n,t,i,j]\
#                   ==0 for w in range(W) for n in range(N) for t in range(T) for i in range(N) for j in range(T) ),\
#         name="cb1")

#     M.addConstrs((q0[w,nn,t]+sum(D_ub[w,i,j]*L_ub[w,nn,t,i,j] for i in range(N) for j in range(T))- \
#                   sum(D_lb[w,i,j]*H_ub[w,nn,t,i,j] for i in range(N) for j in range(T)) \
#                   <=0 for w in range(W) for t in range(1,T)),\
#         name="cb3")

#     M.addConstrs((Q0[w,n,t,i,j]-L_ub[w,n,t,i,j]+H_ub[w,n,t,i,j]\
#                   ==0 for w in range(W) for n in range(N) for t in range(T) for i in range(N) for j in range(T) ),\
#         name="cb4")


#     M.addConstrs((x0[w,n,t]==sum(q0[w,n,m] for m in range(t+1))\
#                   for w in range(W) for n in range(N) for t in range(T)),name='ca0')


#     M.addConstrs((X0[w,n,t,i,tau]==sum(Q0[w,n,m,i,tau] - e(N,T,n,m)[i,tau] for m in range(t+1))\
#                   for w in range(W) for n in range(N) for t in range(T) for i in range(N) for tau in range(T)),name='ca1')


#     M.addConstrs((sum(y[t,r] for r in range(R)) == 1 for t in range(1,T)),name='ca2')
#     M.addConstrs((y[0,r] == 0 for r in range(R)), name="cy1")

#     M.addConstrs((Q0[w,n,t,i,l] == 0 \
#                   for w in range(W) for n in range(N) for t in range(T) \
#                   for l in range(T) if l>=t for i in range(N)),\
#             name="cg2")


#     if valid_flag:
#         for n in range(N):
#             M.addConstrs((sum(H[r][n]*y[tau,r] for r in range(R) for tau in range(t,t+valid_inequality_parameter[n])) 
#                       >= 1 for t in range(1,T-valid_inequality_parameter[n]+1)),
#             name="h0")

#     M.write("asdf.lp")    


#     M.optimize()

#     obje = M.getObjective().getValue()

#     solve_time = time.perf_counter()-start


#     y_t = np.zeros((T,R))
#     for t in range(T):
#         for r in range(R):
#             y_t[t,r] = y[t,r].x

#     alpha_t = np.zeros((N,T))
#     for n in range(N):
#         for t in range(T):
#             alpha_t[n,t] = alpha[n,t].x

#     if W == 1:
#         q_t = {w: np.zeros((N,T,N_test[w])) for w in range(len(D_test))}
#         x_t = {w: np.zeros((N,T,N_test[w])) for w in range(len(D_test))}
#         v_t = {w: np.zeros((N,T,N_test[w])) for w in range(len(D_test))}
#         p_t = {w: np.zeros((N,T,N_test[w])) for w in range(len(D_test))}                            
#         for w in range(len(D_test)):
#             for n in range(N):
#                 for t in range(T):
#                     for i in range(N_test[w]):
#                         q_t[w][n,t,i]=q0[0,n,t].x + sum(Q0[0,n,t,nn,tt].x*D_test[w][nn,tt,i] for nn in range(N) for tt in range(T))
#                         x_t[w][n,t,i]=x0[0,n,t].x + sum(X0[0,n,t,nn,tt].x*D_test[w][nn,tt,i] for nn in range(N) for tt in range(T))               
#                         v_t[w][n,t,i]=max(x_t[w][n,t,i] - tau_ub[n,t], tau_lb[n,t] - x_t[w][n,t,i], 0)            
#                         p_t[w][n,t,i]=abs(v_t[w][n,t,i])>=0.0001 
#     else:                           
#         q_t = {w: np.zeros((N,T,N_test[w])) for w in range(W)}
#         x_t = {w: np.zeros((N,T,N_test[w])) for w in range(W)}
#         v_t = {w: np.zeros((N,T,N_test[w])) for w in range(W)}
#         p_t = {w: np.zeros((N,T,N_test[w])) for w in range(W)}                            
#         for w in range(W):
#             for n in range(N):
#                 for t in range(T):
#                     for i in range(N_test[w]):
#                         q_t[w][n,t,i]=q0[w,n,t].x + sum(Q0[w,n,t,nn,tt].x*D_test[w][nn,tt,i] for nn in range(N) for tt in range(T)) 
#                         x_t[w][n,t,i]=x0[w,n,t].x + sum(X0[w,n,t,nn,tt].x*D_test[w][nn,tt,i] for nn in range(N) for tt in range(T))
#                         v_t[w][n,t,i]=max(x_t[w][n,t,i] - tau_ub[n,t], tau_lb[n,t] - x_t[w][n,t,i], 0)            
#                         p_t[w][n,t,i]=abs(v_t[w][n,t,i])>=0.0001 

#     return obje,alpha_t,y_t,q_t,x_t,v_t,p_t,solve_time

# obje_L,alpha_t_L,y_t_L,q_t_L,x_t_L,v_t_L,p_t_L,solve_time_L\
#     = IRP.single_Scenario_ELDR_calculate(\
#         1,W,N,2,R,P,H,K,a,b,K_l,o,p,tau_lb,tau_ub,D_lb,D_ub,E1,E2,e1,e2,\
#         MM,N_test,D_test,logTo_flag,valid_inequality_parameter,valid_flag=False)
