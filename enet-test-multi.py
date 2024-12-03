import numpy as np
from sdelearn import *
import sympy as sym
import pandas as pd

from time import time

import matplotlib.pyplot as plt
import dill

from scipy.linalg import sqrtm



def corplot(m):
    plt.figure(figsize=(8, 6))
    plt.imshow(m, cmap='coolwarm', interpolation='none', vmin=-1, vmax=1)
    plt.xticks(ticks=np.arange(m.shape[1]), labels=np.arange(0, m.shape[1]))
    plt.yticks(ticks=np.arange(m.shape[0]), labels=np.arange(0, m.shape[0]))
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.show()

def pathplot(lasso, idx):
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(lasso.penalty), lasso.est_path[:, idx])
    plt.show()



# # MODEL SET UP -------------------------------------------------------------------------------------------------------
# n_var = 3
# # declare symbols
# mu = [sym.symbols('mu{0}'.format(i)) for i in range(n_var)]
# theta_dr = [sym.symbols('theta_dr{0}_{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]
# #theta_di = [sym.symbols('theta_di{0}'.format(i)) for i in range(n_var)]
# theta_di = [sym.symbols('theta_di{0}_{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]

# state_var = [sym.symbols('x{0}'.format(i)) for i in range(n_var)]

# all_param = mu + theta_dr + theta_di

# # set drift and diffusion
# # w_it = iter(w_dr)
# # w_arr = np.array([[next(w_it) if i != j else 1 for j in range(n_var)] for i in range(n_var)])

# # b_expr0 = np.array(mu) - w_arr * np.array(theta_dr).reshape((n_var, n_var)) @ np.array(state_var)

# # linear multivariate
# b_expr = np.array(mu) - np.array(theta_dr).reshape((n_var, n_var)) @ np.array(state_var)


# # sqrt model
# #A_expr = np.diag([sym.sqrt(theta_di[i] + state_var[i] ** 2) for i in range(n_var)])

# A_expr = np.array(theta_di).reshape(n_var, n_var)

# # create SDE
# sde = Sde(sampling=SdeSampling(initial=0, terminal=20, delta=0.002),
#           model=SdeModel(b_expr, A_expr, state_var=[s.name for s in state_var]))
# print(sde)

# # CASE 1: INDUCE CORRELATION BY MAKING PARAMETERS SIMILAR

# A_par = np.array([[1, 1.1, 0.9], [1.1, 1, 0.9], [0.9, 1.1, 1]]).flatten()
# b_par = np.eye(n_var).flatten()

# mu_par = np.array([0] * n_var)

# truep = dict(zip([s.name for s in all_param], np.concatenate([mu_par, b_par, A_par])))

# sde.simulate(param=truep, x0=np.array([2] * n_var))
# sde.plot()

# box_width = 100

# d_id = n_var*(1 + n_var) + np.ravel_multi_index(np.diag_indices(n_var), (n_var, n_var))
# # create general bounds tuple
# bounds = np.array([(-0.5 * box_width, 0.5 * box_width)] * len(sde.model.param))
# # replace bounds for diagonal elements drift and diffusion
# bounds[:, 0] = 0
# # add small random noise
# bounds += 0.001 * np.random.rand(len(sde.model.param) * 2).reshape(len(sde.model.param), 2)

# # random starting point
# startp = dict(zip([s for s in sde.model.param], list(map(lambda x: np.random.beta(1, 10) * x[1], bounds))))


# qmle = Qmle(sde)
# qmle.fit(startp, method='TNC', two_step=True, bounds=bounds, options = {'maxfun': 1000})


# for (k, v) in qmle.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))

# np.linalg.eigvalsh(qmle.optim_info['hess'])
# hcorr = np.round(np.corrcoef(qmle.optim_info['hess']), 3)
# qmle.loss(qmle.est) # good: -1.812694921875


# r = sqrtm(qmle.optim_info['hess'])
# rcorr = np.corrcoef(r)
# corplot(rcorr)

# enet = AdaElasticNet(sde, qmle, alpha=0.25, delta=2, n_pen=100)
# lasso = AdaLasso(sde, qmle, delta=2, n_pen=100)
# lasso.fit(epsilon=1e-8, opt_alg='fista')
# lasso.plot()
# pathplot(lasso, [3,6,9])
# pathplot(lasso, [3,4,5])
# pathplot(lasso, [12, 18])
# pathplot(lasso, [13, 19])
# pathplot(lasso, [14, 20])

# # turns out that negatively correlated columns are paired, positively correlated columns are
# # selected (?!)

# enet.fit(epsilon=1e-8, opt_alg='fista', cv=0.1)
# enet.coef(enet.lambda_min)
# enet.plot()






# # METHOD 2: ADD COPY OF A VAR -------------------------------------------------------------------------------------------------------
# n_var = 2
# # declare symbols
# mu = [sym.symbols('mu{0}'.format(i)) for i in range(n_var)]
# theta_dr = [sym.symbols('theta_dr{0}_{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]
# #theta_di = [sym.symbols('theta_di{0}'.format(i)) for i in range(n_var)]
# theta_di = [sym.symbols('theta_di{0}_{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]

# state_var = [sym.symbols('x{0}'.format(i)) for i in range(n_var)]

# all_param = mu + theta_dr + theta_di

# # set drift and diffusion
# # w_it = iter(w_dr)
# # w_arr = np.array([[next(w_it) if i != j else 1 for j in range(n_var)] for i in range(n_var)])

# # b_expr0 = np.array(mu) - w_arr * np.array(theta_dr).reshape((n_var, n_var)) @ np.array(state_var)

# # linear multivariate
# b_expr = np.array(mu) - np.array(theta_dr).reshape((n_var, n_var)) @ np.array(state_var)


# # sqrt model
# #A_expr = np.diag([sym.sqrt(theta_di[i] + state_var[i] ** 2) for i in range(n_var)])

# A_expr = np.array(theta_di).reshape(n_var, n_var)
# np.fill_diagonal(A_expr, [1/sym.sqrt(np.array(theta_di).reshape((n_var, n_var))[i,i] + state_var[i] ** 2) for i in range(n_var)])
# # create SDE
# sde = Sde(sampling=SdeSampling(initial=0, terminal=10, delta=0.002),
#           model=SdeModel(b_expr, A_expr, state_var=[s.name for s in state_var]))
# print(sde)

# # CASE 1: INDUCE CORRELATION BY MAKING PARAMETERS SIMILAR

# A_par = np.eye(n_var).flatten()
# b_par = np.eye(n_var).flatten()

# mu_par = np.array([0] * n_var)

# truep = dict(zip([s.name for s in all_param], np.concatenate([mu_par, b_par, A_par])))

# sde.simulate(param=truep, x0=np.array([2] * n_var))
# sde.plot()




# # initial estimate - do not run -----
# box_width = 100

# d_id = n_var*(1 + n_var) + np.ravel_multi_index(np.diag_indices(n_var), (n_var, n_var))
# # create general bounds tuple
# bounds = np.array([(-0.5 * box_width, 0.5 * box_width)] * len(sde.model.param))
# # replace bounds for diagonal elements drift and diffusion
# bounds[:, 0] = 0
# # add small random noise
# bounds += 0.001 * np.random.rand(len(sde.model.param) * 2).reshape(len(sde.model.param), 2)

# # random starting point
# startp = dict(zip([s for s in sde.model.param], list(map(lambda x: np.random.beta(1, 10) * x[1], bounds))))


# qmle = Qmle(sde)
# qmle.fit(startp, method='TNC', two_step=True, bounds=bounds, options = {'maxfun': 1000})
# print(qmle.loss(qmle.est))

# for (k, v) in qmle.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))

# np.linalg.eigvalsh(qmle.optim_info['hess'][6:10, 6:10])
# np.linalg.eigvalsh(qmle.optim_info['hess'][0:10, 0:10])

# np.round(np.corrcoef(qmle.optim_info['hess']), 3)


# enet = AdaElasticNet(sde, qmle, alpha=0.25, delta=2, n_pen=100)
# lasso = AdaLasso(sde, qmle, delta=2, n_pen=100)
# lasso.fit(epsilon=1e-8, opt_alg='fista')
# lasso.plot()
# enet.fit(epsilon=1e-8, opt_alg='fista', cv=0.2)
# enet.coef(enet.lambda_min)
# enet.plot()
# enet.val_loss

# import matplotlib.pyplot as plt
# plt.plot(np.arange(78), enet.val_loss[0:78], color="red")
# plt.show()

# qmle.loss(enet.coef(enet.lambda_opt))
# qmle.loss(qmle.est)








# # pass to larger model ---------------------------------------------------------

# n_var = 3
# # declare symbols
# mu = [sym.symbols('mu{0}'.format(i)) for i in range(n_var)]
# theta_dr = [sym.symbols('theta_dr{0}_{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]
# #theta_di = [sym.symbols('theta_di{0}'.format(i)) for i in range(n_var)]
# theta_di = [sym.symbols('theta_di{0}_{1}'.format(i, j)) for i in range(n_var) for j in range(n_var)]
# state_var = [sym.symbols('x{0}'.format(i)) for i in range(n_var)]

# all_param = mu + theta_dr + theta_di

# # set drift and diffusion
# # w_it = iter(w_dr)
# # w_arr = np.array([[next(w_it) if i != j else 1 for j in range(n_var)] for i in range(n_var)])

# # b_expr0 = np.array(mu) - w_arr * np.array(theta_dr).reshape((n_var, n_var)) @ np.array(state_var)

# # linear multivariate
# b_expr = np.array(mu) - np.array(theta_dr).reshape((n_var, n_var)) @ np.array(state_var)


# # sqrt model
# #A_expr = np.diag([sym.sqrt(theta_di[i] + state_var[i] ** 2) for i in range(n_var)])

# A_expr = np.array(theta_di).reshape(n_var, n_var)
# np.fill_diagonal(A_expr, [1/sym.sqrt(np.array(theta_di).reshape((n_var, n_var))[i,i] + state_var[i] ** 2) for i in range(n_var)])

# # make enlarged data
# data = np.copy(sde.data.data.to_numpy())
# data = np.column_stack((data[:, 0:2], data[:,1] + 0.01 * np.random.randn(data[:,1].shape[0])))
# # create SDE
# sde = Sde(sampling=SdeSampling(initial=0, terminal=10, delta=0.002),
#           model=SdeModel(b_expr, A_expr, state_var=[s.name for s in state_var]),
#           data=SdeData(data))
# print(sde)

# sde.plot()

# box_width = 100

# d_id = n_var*(1 + n_var) + np.ravel_multi_index(np.diag_indices(n_var), (n_var, n_var))
# # create general bounds tuple
# bounds = np.array([(-0.5 * box_width, 0.5 * box_width)] * len(sde.model.param))
# # replace bounds for diagonal elements drift and diffusion
# bounds[:, 0] = 0
# # add small random noise
# bounds += 0.001 * np.random.rand(len(sde.model.param) * 2).reshape(len(sde.model.param), 2)

# # random starting point
# startp = dict(zip([s for s in sde.model.param], list(map(lambda x: np.random.beta(1, 10) * x[1], bounds))))


# qmle = Qmle(sde)
# qmle.fit(startp, method='TNC', two_step=True, bounds=bounds, options = {'maxfun': 1000})
# print(qmle.loss(qmle.est))

# for (k, v) in qmle.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))

# np.linalg.eigvalsh(qmle.optim_info['hess'][6:10, 6:10])
# np.linalg.eigvalsh(qmle.optim_info['hess'][0:10, 0:10])

# np.round(np.corrcoef(qmle.optim_info['hess']), 3)




# enet = AdaElasticNet(sde, qmle, alpha=0.25, delta=2, n_pen=100)
# lasso = AdaLasso(sde, qmle, delta=2, n_pen=100)
# lasso.fit(epsilon=1e-8, opt_alg='fista')
# lasso.plot()
# enet.fit(epsilon=1e-8, opt_alg='fista')
# enet.plot()



# r = sqrtm(qmle.optim_info['hess'])
# rcorr = np.corrcoef(qmle.optim_info['hess'])
# corplot(rcorr)



# pathplot(lasso, [3, 5])
# pathplot(enet, [3, 5])

# pathplot(enet, [6, 7, 8])
# pathplot(lasso, [6,7])
# pathplot(enet, [6,7])



# pathplot(lasso, [12, 18])
# pathplot(lasso, [16, 20])
# pathplot(lasso, [13, 20])
# pathplot(enet, [16, 20])
# pathplot(enet, [13, 20])


# hess_corr = np.corrcoef(qmle.optim_info['hess'])
# hess_corr[17,19]

# with open('enet-corr.pkl', 'wb') as outp:
#     dill.dump(enet, outp)
# with open('lasso-corr.pkl', 'wb') as outp:
#     dill.dump(lasso, outp)



# with open('enet-corr.pkl', 'rb') as inp:
#     enet = dill.load(inp)
# with open('lasso-corr.pkl', 'rb') as inp:
#     lasso = dill.load(inp)

# qmle = lasso.base_est
# hess_corr = np.corrcoef(qmle.optim_info['hess'])




























# # METHOD 3: STOCHASTIC REGRESSION  -----------------------------------------------------------------------------------------------------

# # simulate and return sde with some diffusion parameters fixed - old version 2 vars
# def sim_step(ret_eval=True, est_opt=True):

#     n_var = 3
#     # declare symbols
#     beta = [s for s in sym.symbols('beta00, beta01, beta11, beta22')]
#     alpha = [sym.symbols('alpha{0}'.format(i)) for i in range(n_var+1)]

#     #theta_di = [sym.symbols('theta_di{0}'.format(i)) for i in range(n_var)]
#     sigma = sym.symbols('sigma')
#     theta = np.array([[sym.symbols('theta_di{0}{1}'.format(i,j)) for j in range(1, n_var)] for i in range(1, n_var)])


#     x1, x2, y = sym.symbols('x1 x2 y')

#     state_var = [x1, x2, y]

#     b_expr = [
#         - alpha[1]*y + alpha[2] * x1 + alpha[3] * x2,
#         beta[0] - beta[2] * x1,
#         beta[1] - beta[3] * x2,
#     ]


#     A_expr = np.block([
#         [sigma, np.zeros((1, 2))],
#         [np.zeros((2, 1)), theta]
#     ])

#     # create SDE
#     sde = Sde(sampling=SdeSampling(initial=0, terminal=25, delta=0.1),
#               model=SdeModel(b_expr, A_expr, state_var=['y', 'x1', 'x2']))
#     #print(sde)

#     theta_val = sqrtm([[1, 0.9], [0.9, 1]])
#     truep = {'alpha1': 1, 'alpha2': 1, 'alpha3': 1,
#              'beta00': 0, 'beta01': 0, 'beta11': 1, 'beta22': 1,
#              'sigma': 1,
#              'theta_di11': theta_val[0,0], 'theta_di12': theta_val[0,1], 'theta_di21': theta_val[1,0], 'theta_di22': theta_val[1,1]}

#     sde.simulate(param=truep, x0=[0, 0, 0])

#     #sde.plot()

#     fix_par = {'theta_di11': theta_val[0,0], 'theta_di12': theta_val[0,1],
#                'theta_di21': theta_val[1,0], 'theta_di22': theta_val[1,1]}

#     b_expr2 = [b_expr[i].subs(fix_par) for i in range(n_var)]
#     A_expr2 = np.array([[A_expr[i,j] if A_expr[i,j] == 0 else A_expr[i,j].subs(fix_par) for j in range(n_var)] for i in range(n_var)])


#     # create SDE
#     sde2 = Sde(sampling=sde.sampling,
#               model=SdeModel(b_expr2, A_expr2, state_var=['y', 'x1', 'x2']),
#               data = sde.data)
#     #print(sde2)
#     #sde2.plot()

#     truep2 = {k: truep.get(k) for k in sde2.model.param}

#     box_width = 100

#     # create general bounds tuple
#     bounds = np.array([(-0.5 * box_width, 0.5 * box_width)] * len(sde2.model.param))
#     # replace bounds for diagonal elements drift and diffusion
#     bounds[:, 0] = 0
#     # add small random noise
#     bounds += 0.001 * np.random.rand(len(sde2.model.param) * 2).reshape(len(sde2.model.param), 2)

#     # random starting point
#     startp = dict(zip([s for s in sde2.model.param], list(map(lambda x: np.random.beta(1, 10) * x[1], bounds))))


#     qmle = Qmle(sde2)
#     qmle.fit(startp, method='TNC', two_step=True, bounds=bounds, options={'maxfun': 1000})


#     enet = AdaElasticNet(sde2, qmle, alpha=0.1, delta=2, n_pen=100)
#     lasso = AdaLasso(sde2, qmle, delta=2, n_pen=100)
#     lasso.fit(epsilon=1e-6, opt_alg='fista', cv=None, nfolds=10, cv_metric='loss', aic_k=1)
#     enet.fit(epsilon=1e-6, opt_alg='fista', cv=None, nfolds=10, cv_metric='loss', aic_k=1)

#     if ret_eval:
#         reg_par = ['alpha2', 'alpha3'] #regression params for model evaluation, can be generalized
#         return mod_eval(lasso, truep2, reg_par, est_opt) + mod_eval(enet, truep2, reg_par, est_opt)
#     else:
#         return qmle, lasso, enet, truep2




# # compute selection accuracy (TP + TN)/npar and check all the correlated variables params (string list reg_par) are selected
# # est_opt uses lambda_opt, otherwise lambda_min
# def mod_eval(mod, truep, reg_par, est_opt=True):
#     true0 = np.array(list(truep.values())) == 0
#     if est_opt:
#         est = mod.coef(mod.lambda_opt)
#     else:
#         est = mod.est
#     est0 = np.array(list(est.values())) == 0
#     est1 = np.array([est.get(k) for k in reg_par])
#     return np.sum(np.equal(true0, est0))/len(true0), np.all(est1 != 0)

# B = 100
# lacc = np.empty(B)
# lsel = np.empty(B)
# eacc = np.empty(B)
# esel = np.empty(B)

# for i in range(B):
#     lacc[i], lsel[i], eacc[i], esel[i] = sim_step(est_opt=True)
#     if (i+1) % (B//100*2) == 0 or i==0:
#         print(f"[{'='*((i+1)//(B//100*2)):<50}] {int((i+1)/B*100)}% ", end='\r')

# np.mean(lacc)
# np.mean(lsel)
# np.mean(eacc)
# np.mean(esel)

# # RESULTS
# # n = 250 (delta = 0.1, T = 25) [lambda_min]
# # lasso accuracy: 0.88, lasso selection: 0.27, enet accuracy: 0.90, enet selection: 0.73 (min)
# # lasso accuracy: 0.89, lasso selection: 0.19, enet accuracy: 0.93, enet selection: 0.70 (opt)
# # n = 1000 (delta = 0.05, T = 50) [lambda_min]
# # lasso accuracy: 0.92, lasso selection: 0.6, enet accuracy: 0.93, enet selection: 0.89 (min)
# # n = 10000 (delta = 0.01, T = 100) [lambda_min]
# # lasso accuracy: 0.97, lasso selection: 0.87, enet accuracy: 0.97, enet selection: 0.98



# # current best config: delta 0.1, terminal 50 : 0.88 - 0.97


# qmle, lasso, enet, truep = sim_step(ret_eval=False)

# for (k, v) in qmle.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))


# for k, v in lasso.coef(lasso.lambda_opt).items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))


# for k, v in enet.coef(enet.lambda_opt).items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))

# for k, v in enet.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))

# for k, v in lasso.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))



# [enet.est.get(k) for k in ['alpha2', 'alpha3']]
# mod_eval(enet, truep,  ['alpha2', 'alpha3'])


# TODO: MODEL EVALUATION
# PERCENTAGE OF TIMES BOTH VARIABLES WERE KEPT VS SELECTION ACCURACY















# STOCHASTIC REGRESSION 2: INCREASE NUMBER OF PREDICTORS
# simulate and return sde with some diffusion parameters fixed - old version 2 vars
def sim_step(ret_eval=True, est_opt=True, n_var=5, enet_mix=0.5):

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
    sde = Sde(sampling=SdeSampling(initial=0, terminal=25, delta=0.1),
              model=SdeModel(b_expr, A_expr, state_var=state_var))
    #print(sde)

    # assign parameter values
    alpha_val = [1]*((n_var-1)//2) + [0]*((n_var-1) - (n_var-1)//2) + [1]
    #alpha_val = [1]*(n_var)
    alpha_par = dict(zip([s.name for s in alpha[1:]], alpha_val))

    beta_val = [0,1]*(n_var-1)
    beta_par = dict(zip([s.name for s in beta], beta_val))

    cor_mat = np.identity(n_var - 1)
    rho = 0.5
    for j in range(1, n_var - 1):
        cor_mat += np.diag([rho ** j] * (n_var - 1 - j), j) + np.diag([rho ** j] * (n_var - 1 - j), -j)
    theta_val = sqrtm(cor_mat)
    theta_par = {theta[i,j].name: theta_val[i,j] for i in range(n_var-1) for j in range(n_var-1)}

    sigma_par = {'sigma': 1}

    truep = {**alpha_par, **beta_par, **theta_par, **sigma_par}

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

    box_width = 100

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


    enet = AdaElasticNet(sde2, qmle, alpha=enet_mix, delta=0.1, n_pen=100)
    lasso = AdaLasso(sde2, qmle, delta=0.1, n_pen=100)
    lasso.fit(epsilon=1e-6, opt_alg='fista', cv=None, nfolds=10, cv_metric='loss', aic_k=1)
    enet.fit(epsilon=1e-6, opt_alg='fista', cv=None, nfolds=10, cv_metric='loss', aic_k=1)

    if ret_eval:
        reg_par = [s.name for s in alpha[1:(n_var)]] #regression params for model evaluation, can be generalized
        return mod_eval(lasso, truep2, reg_par, est_opt) + mod_eval(enet, truep2, reg_par, est_opt)
    else:
        return qmle, lasso, enet, truep2




# compute selection accuracy (TP + TN)/npar and check all the correlated variables params (string list reg_par) are selected
# est_opt uses lambda_opt, otherwise lambda_min
def mod_eval(mod, truep, reg_par, est_opt=True):
    true0 = np.array(list(truep.values())) == 0
    if est_opt:
        est = mod.coef(mod.lambda_opt)
    else:
        est = mod.est
    est0 = np.array(list(est.values())) == 0
    est1 = np.array([est.get(k) for k in reg_par])
    return np.sum(np.equal(true0, est0))/len(true0), np.all(est1 != 0)

B = 10
lacc = np.empty(B)
lsel = np.empty(B)
eacc = np.empty(B)
esel = np.empty(B)



for i in range(B):
    lacc[i], lsel[i], eacc[i], esel[i] = sim_step(est_opt=True, n_var=10, enet_mix=0.1)
    if (i+1) % (B//100*2) == 0 or i==0:
        print(f"[{'='*((i+1)//(B//100*2)):<50}] {int((i+1)/B*100)}% ", end='\r')

np.mean(lacc)
np.mean(lsel)
np.mean(eacc)
np.mean(esel)



# qmle, lasso, enet, truep2 = sim_step(ret_eval=False)

# for (k, v) in qmle.est.items():
#     print(k, '  %.5f  ' % v, '  ', truep2.get(k))


# for k, v in lasso.coef(lasso.lambda_opt).items():
#     print(k, '  %.5f  ' % v, '  ', '  %.5f  ' % enet.coef(enet.lambda_opt).get(k), truep.get(k))


# for k, v in enet.coef(enet.lambda_opt).items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))













#
# A_hat = np.array(list(qmle.est.values())[-4:]).reshape(2, 2)
# A_hat@A_hat

li
# lasso.plot()
# enet.plot()
#
#
# plt.figure()
# plt.plot(lasso.val_loss[0:70])
# # plt.figure()
# # plt.plot(np.mean(lasso.val_loss, axis=1)[0:70])
# #
# # plt.figure()
# # plt.plot(np.mean(enet.val_loss, axis=1)[0:70])
#
#
#
# plt.figure()
# plt.plot(enet.val_loss[0:70])

# for k, v in enet.coef(enet.lambda_opt).items():
#     print(k, '  %.5f  ' % v, '  ', truep.get(k))

sde.model.der_foo['b'](*sde.data.original_data[0], **truep)

# foo(a, b, c)
# l = [a0, b0, c0], d={a: a0, b:b0, c:c0}
# foo(a=a0, b=b0, c=c0) = foo(*l) = f(**d)

iqr = np.quantile(enet.val_loss, 0.6, interpolation='midpoint') - np.quantile(enet.val_loss, 0.1, interpolation='midpoint')
lopt = enet.penalty[1:-1][enet.val_loss < np.nanmin(enet.val_loss) + iqr][-1]

for k, v in enet.coef(lopt).items():
    print(k, '  %.5f  ' % v, '  ', truep.get(k))

np.log(lopt)

iqr2 = np.quantile(lasso.val_loss, 0.6, interpolation='midpoint') - np.quantile(lasso.val_loss, 0.1, interpolation='midpoint')
lopt2 = lasso.penalty[1:-1][lasso.val_loss < np.nanmin(lasso.val_loss) + iqr2][-1]

for k, v in lasso.coef(lopt2).items():
    print(k, '  %.5f  ' % v, '  ', truep.get(k))

np.log(lopt2)
r = sqrtm(qmle.optim_info['hess'])
rcorr = np.corrcoef(r)
corplot(rcorr)


pathplot(lasso, [1, 2])
pathplot(enet, [1, 2])

pathplot(enet, [6, 7, 8])
pathplot(lasso, [3, 4])
pathplot(enet, [3, 4])

pathplot(lasso, [4, 5])
pathplot(enet, [4, 5])


pathplot(lasso, [9, 10])
pathplot(enet, [9, 10])