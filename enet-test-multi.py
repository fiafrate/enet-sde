import numpy as np
from sdelearn import *
import sympy as sym
import pandas as pd

from time import time

import matplotlib.pyplot as plt
import dill
from pathos.multiprocessing import ProcessingPool as Pool
import os 
from scipy.linalg import sqrtm

import multiprocess
from multiprocess import Pool, cpu_count
#import dill


# Ensure `dill` is set as the default serializer
#dill.settings['recurse'] = True

# import sys
# sys.setrecursionlimit(1000)

import os



# STOCHASTIC REGRESSION 2: INCREASE NUMBER OF PREDICTORS



def sim_step(ret_eval=True, est_opt=True, n_var=5, 
             sigma_val=1, rho=0.5,
             samp_delta=0.1, samp_term=50, 
             enet_mix=0.5, delta_pen=0.5):

    #n_var = 5
    # declare symbols
    beta = [sym.symbols('beta{0}{1}'.format(i, j)) for i in range(1, n_var) for j in range(2)]
    alpha = [sym.symbols('alpha{0}'.format(i)) for i in range(n_var+1)]

    #theta_di = [sym.symbols('theta_di{0}'.format(i)) for i in range(n_var)]
    sigma = sym.symbols('sigma')
    theta = np.array([[sym.symbols('theta_di{0}{1}'.format(i,j)) for j in range(1, n_var)] for i in range(1, n_var)])


    x_var = [sym.symbols('x{0}'.format(i)) for i in range(1, n_var)]
    y = sym.symbols('y')
    state_var = [s.name for s in  [y] + x_var]

    b_expr = ([- alpha[n_var]*y + np.sum([alpha[j] * x_var[j-1] for j in range(1, n_var)])]
              + [beta[2 * i] - beta[2 * i + 1] * x_var[i] for i in range(n_var - 1)])


    A_expr = np.block([
        [sigma, np.zeros((1, n_var-1))],
        [np.zeros((n_var-1, 1)), theta]
    ])

    # create SDE
    sde = Sde(sampling=SdeSampling(initial=0, terminal=samp_term, delta=samp_delta),
              model=SdeModel(b_expr, A_expr, state_var=state_var))
    #print(sde)

    # assign parameter values
    alpha_val = [1]*((n_var-1)//2) + [0]*((n_var-1) - (n_var-1)//2) + [n_var]
    #alpha_val = [1]*(n_var)
    alpha_par = dict(zip([s.name for s in alpha[1:]], alpha_val))

    beta_val = [0,1]*(n_var-1)
    beta_par = dict(zip([s.name for s in beta], beta_val))

    cor_mat = np.identity(n_var - 1)
    for j in range(1, n_var - 1):
        cor_mat += np.diag([rho ** j] * (n_var - 1 - j), j) + np.diag([rho ** j] * (n_var - 1 - j), -j)
    theta_val = sqrtm(cor_mat)
    theta_par = {theta[i,j].name: theta_val[i,j] for i in range(n_var-1) for j in range(n_var-1)}

    sigma_par = {'sigma': sigma_val}

    truep = {**alpha_par, **beta_par, **theta_par, **sigma_par}

    np.random.seed(os.getpid() + int(time()%10 * 10000))
    sde.simulate(param=truep, x0=[0]*n_var)

    #sde.plot()

    fix_par = {**theta_par, **sigma_par}

    b_expr2 = [b_expr[i].subs(fix_par) for i in range(n_var)]
    A_expr2 = np.array([[A_expr[i,j] if A_expr[i,j] == 0 else A_expr[i,j].subs(fix_par) for j in range(n_var)] for i in range(n_var)])


    # create SDE
    sde2 = Sde(sampling=sde.sampling,
              model=SdeModel(b_expr2, A_expr2, state_var=state_var),
              data = sde.data)
    #print(sde2)
    #sde2.plot()

    truep2 = {k: truep.get(k) for k in sde2.model.param}

    box_width = 99

    # create general bounds tuple
    bounds = np.array([(-0.5 * box_width, 0.5 * box_width)] * len(sde2.model.param))
    # replace bounds for diagonal elements drift and diffusion
    bounds[:, 0] = 0
    # add small random noise
    bounds += 0.001 * np.random.rand(len(sde2.model.param) * 2).reshape(len(sde2.model.param), 2)

    # random starting point
    startp = dict(zip([s for s in sde2.model.param], list(map(lambda x: np.random.beta(1, 10) * x[1], bounds))))


    qmle = Qmle(sde2)
    qmle.fit(startp, method='TNC', two_step=True, bounds=bounds, options={'maxfun': 1000})

    enet = AdaElasticNet(sde2, qmle, alpha=enet_mix, delta=delta_pen, n_pen=100)
    lasso = AdaLasso(sde2, qmle, delta=delta_pen, n_pen=100)
    lasso.fit(epsilon=1e-6, opt_alg='fista', cv=None, nfolds=10, cv_metric='loss', aic_k=1)
    enet.fit(epsilon=1e-6, opt_alg='fista', cv=None, nfolds=10, cv_metric='loss', aic_k=1)

    if ret_eval:
        reg_par = [s.name for s in alpha[1:(n_var)]] #regression params for model evaluation, can be generalized
        eval_lasso = mod_eval(lasso, truep2, reg_par, est_opt)
        eval_enet =  mod_eval(enet, truep2, reg_par, est_opt)
        out = eval_lasso + eval_enet
        return out
    else:
        return qmle, lasso, enet, truep2





# compute selection accuracy (TP + TN)/npar and check all the correlated variables params (string list reg_par) are selected
# est_opt uses lambda_opt, otherwise lambda_min
def mod_eval(mod, truep, reg_par, est_opt=True):
    true0 = np.array(list(truep.values())) == 0
    true1 = np.array([truep.get(k) for k in reg_par]) != 0 #non-zero variables in reg
    if est_opt:
        est = mod.coef(mod.lambda_opt)
    else:
        est = mod.est
    est0 = np.array(list(est.values())) == 0
    est1 = np.array([est.get(k) for k in reg_par])[true1] #were all non-zero in reg included?
    acc = np.sum(np.equal(true0, est0))/len(true0)
    sel = np.all(est1 != 0) 
    return acc, sel


# batched version
# args: list of dicts
def wrapped_sim(args):
    return [sim_step(**arg) for arg in args]


if __name__ == "__main__":
    start_time = time()

    num_processes = int(os.getenv("SLURM_CPUS_PER_TASK", default=1))

    n_batch = 10*num_processes
    batch_size = 10
    B = n_batch * batch_size

    lacc = np.empty(B)
    lsel = np.empty(B)
    eacc = np.empty(B)
    esel = np.empty(B)


    sim_config = {
        'n_var': 10, 
        'sigma_val': 1, 
        'rho': 0.8,
        'samp_delta':0.1, 
        'samp_term': 50, 
        'enet_mix':0.5, 
        'delta_pen':1,
    }


    # non parallelized for loop, n_batch and batch_size are not really used
    # for i in range(B):
    #     lacc[i], lsel[i], eacc[i], esel[i] = sim_step(est_opt=False, **sim_config)
    # #     if (i+1) % batch_size == 0 or i==0:
    # #         print(f"[{'='*((i+1)//batch_size):<50}] {int((i+1)/B*100)}% ", end='\r')

    print(num_processes)
    #multiprocess.set_start_method("spawn")
    with Pool(processes=num_processes) as executor:
        results = list(executor.map(wrapped_sim, [[sim_config]*batch_size]*n_batch))

    num_results = np.array([x for item in results for x in item])

    lacc = num_results[:,0]
    lsel = num_results[:,1]
    eacc = num_results[:,2]
    esel = num_results[:,3]

    # data_list = []
    # for i in range(B):
    #     data_list.append(results[i][0].sde.data.original_data)



    print(pd.DataFrame({'lasso acc': [np.mean(lacc)], 'lasso sel': [np.mean(lsel)], 'enet acc': [np.mean(eacc)], 'enet sel': [np.mean(esel)]}))


    # TESTS:
    # INCREASE SIGMA: no
    # INCREASE RHO: no
    # REDUCE ENET_MIX: ok


    # LOCAL TEST
    # qmle, lasso, enet, truep2 = sim_step(ret_eval=False, **sim_config)

    # qmle.sde.plot()
    # for (k, v) in qmle.est.items():
    #     print(k, '  %.5f  ' % v, '  ', truep2.get(k))

    # for k, v in lasso.coef(lasso.lambda_opt).items():
    #     print(k, '  %.8f  ' % v, '  ', '  %.8f  ' % enet.coef(enet.lambda_opt).get(k), '  %.5f  ' % qmle.est.get(k), truep2.get(k))

    # reg_par = ['alpha{0}'.format(i) for i in range(1,sim_config['n_var'])]
    # mod_eval(mod=lasso, truep=truep2, reg_par=reg_par)
    # mod_eval(mod=enet, truep=truep2, reg_par=reg_par)
    # for k, v in enet.coef(enet.lambda_opt).items():
    #     print(k, '  %.5f  ' % v, '  ', truep.get(k))

    print(time() - start_time)
