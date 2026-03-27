from itertools import combinations
import numpy as np
import pandas as pd
from scipy.io import loadmat
import warnings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LF = loadmat(os.path.join(BASE_DIR, 'LFND.mat'))['LFND']

def multi_objective_lichtenberg_algorithm(fhandle, n_objectives, d, pop, LB, UB, ref, n_iter, ngrid, Nr, IntCon, fnonlin):    
    POS = np.zeros((pop, d))
    POS_fit = np.zeros((pop, n_objectives))

    for i in range(pop):
        POS[i, :] = LB + (UB - LB) * np.random.rand(1, d)

        if IntCon != 0:
            POS[:, IntCon] = np.round(POS[:, IntCon])
        POS_fit[i, :] = Fun(fhandle, fnonlin, POS[i, :])
    
    PBEST = np.copy(POS)
    PBEST_fit = np.copy(POS_fit)
    DOMINATED = checkDomination(POS_fit)
          
    REP = {
        'pos': POS[~DOMINATED, :],
        'pos_fit': POS_fit[~DOMINATED, :]
    }
    REP = updateGrid(REP, ngrid)

    LF = loadmat('LFND.mat')['LFND']

    iteration = 0
    for t in range(n_iter):
        iteration += 1        
        
        random_index =  np.random.choice(REP['pos'].shape[0], 1, replace=False)[0]
        
        x_start = REP['pos'][random_index, :]

        scale_factor = 1.2 * np.random.rand()
                
        X = LA_points(LF, LB, UB, x_start, scale_factor, d)
        
        if ref != 0:            
            X_local = LA_points(LF, LB * ref, UB * ref, x_start, scale_factor, d)

        for i in range(pop):
            if ref != 0:
                pop1 = round(0.4 * pop)
                pop2 = pop - pop1
                S_global = X[np.random.permutation(len(X))[:pop2], :]
                S_ref = X_local[np.random.permutation(len(X_local))[:pop1], :]
                                
                POS = np.vstack((S_global, S_ref))
            else:
                POS = X[np.random.permutation(len(X))[:pop], :]     

        if IntCon != 0:
            for i in IntCon:
                for j in range(POS.shape[0]):
                    POS[j, i] = round(POS[j, i])
            
        for kk in range(POS.shape[0]):
            index1 = np.where(POS[kk, :] > UB)[0]
            index2 = np.where(POS[kk, :] < LB)[0]
            POS[kk, index1] = UB[index1]
            POS[kk, index2] = LB[index2]

        for j in range(POS.shape[0]):
            POS_fit[j, :] = Fun(fhandle, fnonlin, POS[j, :])

        REP = updateRepository(REP, POS, POS_fit, ngrid)

        if REP['pos'].shape[0] > Nr:
            REP = deleteFromRepository(REP, REP['pos'].shape[0] - Nr, ngrid)
       
        pos_best = dominates(POS_fit, PBEST_fit)
        best_pos = ~dominates(PBEST_fit, POS_fit)
        best_pos[np.random.rand(pop) >= 0.5] = 0
                
        if np.sum(pos_best) > 1:
            PBEST_fit[pos_best, :] = POS_fit[pos_best, :]
            PBEST[pos_best, :] = POS[pos_best, :]

        if np.sum(best_pos) > 1:
            PBEST_fit[best_pos, :] = POS_fit[best_pos, :]
            PBEST[best_pos, :] = POS[best_pos, :]

    XPF = REP['pos']
    FVALPF = REP['pos_fit']

    return XPF, FVALPF

def Fun(fhandle, fnonlin, u):
    z = fhandle(u)
    z = z + getconstraints(fnonlin, u)
    return z

def getH(g):
    return 0 if g <= 0 else 1

def geteqH(g):
    return 0 if g == 0 else 1

def getconstraints(fnonlin, u):
    PEN = 1e15
    lam = PEN
    lameq = PEN
    g, geq = fnonlin(u)    
    g_np = np.array(g)
    geq_np = np.array(geq)
    Z_ineq = lam * np.sum(g_np**2 * (g_np > 0))
    Z_eq = lameq * np.sum(geq_np**2)    
    return Z_ineq + Z_eq

def updateRepository(REP, POS, POS_fit, ngrid):
    
    novas_posicoes_nao_dominadas = ~checkDomination(POS_fit)
    REP['pos'] = np.vstack((REP['pos'], POS[novas_posicoes_nao_dominadas, :]))
    REP['pos_fit'] = np.vstack((REP['pos_fit'], POS_fit[novas_posicoes_nao_dominadas, :]))

    repositorio_dominadas = checkDomination(REP['pos_fit'])
    REP['pos'] = REP['pos'][~repositorio_dominadas, :]
    REP['pos_fit'] = REP['pos_fit'][~repositorio_dominadas, :]

    REP = updateGrid(REP, ngrid)
    return REP

def checkDomination(fitness):
    num_particulas = fitness.shape[0]
    vetor_dominacao = np.zeros(num_particulas, dtype=bool)

    if num_particulas < 2:
        return vetor_dominacao

    pares_comparacao = np.array(list(combinations(range(num_particulas), 2)))

    if pares_comparacao.size == 0:
        return vetor_dominacao

    indices_i = np.hstack((pares_comparacao[:, 0], pares_comparacao[:, 1]))
    indices_j = np.hstack((pares_comparacao[:, 1], pares_comparacao[:, 0]))

    indices_dominacao = dominates(fitness[indices_i, :], fitness[indices_j, :])

    particulas_dominadas_globais = indices_j[indices_dominacao]

    if particulas_dominadas_globais.size > 0:
        vetor_dominacao[np.unique(particulas_dominadas_globais)] = True

    return vetor_dominacao

def dominates(x, y):    
    return np.all(x <= y, axis=1) & np.any(x < y, axis=1)

def updateGrid(REP, ngrid):    
    ndim = REP['pos_fit'].shape[1]
        
    if REP['pos_fit'].shape[0] == 1:
        REP['hypercube_limits'] = np.tile(REP['pos_fit'][0], (ngrid + 1, 1))    
        REP['grid_idx'] = np.array([1]) 
        REP['grid_subidx'] = np.ones((1, ndim), dtype=int)
        REP['quality'] = np.array([[1, 10/1]])
        return REP
    
    REP['hypercube_limits'] = np.zeros((ngrid + 1, ndim))
    for dim in range(ndim):
        min_val = np.min(REP['pos_fit'][:, dim])
        max_val = np.max(REP['pos_fit'][:, dim])
        if max_val == min_val:
            REP['hypercube_limits'][:, dim] = np.linspace(min_val, min_val + 1e-9, ngrid + 1)
        else:
            REP['hypercube_limits'][:, dim] = np.linspace(min_val, max_val, ngrid + 1)
    
    npar = REP['pos_fit'].shape[0]
    REP['grid_idx'] = np.zeros(npar, dtype=int)
    REP['grid_subidx'] = np.zeros((npar, ndim), dtype=int)

    for n in range(npar):
        sub_indices = np.zeros(ndim, dtype=int)
        for d in range(ndim):                        
            idx = np.searchsorted(REP['hypercube_limits'][:, d], REP['pos_fit'][n, d], side='right')            
            sub_indices[d] = max(1, min(idx, ngrid)) 
        
        REP['grid_subidx'][n, :] = sub_indices                
        REP['grid_idx'][n] = np.ravel_multi_index(sub_indices - 1, dims=tuple([ngrid]*ndim)) + 1
    
    unique_ids, counts = np.unique(REP['grid_idx'], return_counts=True)
    REP['quality'] = np.zeros((len(unique_ids), 2))
    REP['quality'][:, 0] = unique_ids
    REP['quality'][:, 1] = 10 / counts

    return REP


def deleteFromRepository(REP, n_extra, ngrid):
    num_individuals = REP['pos'].shape[0]
    num_objectives = REP['pos_fit'].shape[1]    
    crowding = np.zeros(num_individuals)    
    for m in range(num_objectives):        
        current_objective_fits = REP['pos_fit'][:, m]

        idx_sorted_matlab = np.argsort(current_objective_fits, kind='mergesort')
        sorted_fits = current_objective_fits[idx_sorted_matlab]

        m_up_sorted = np.concatenate((sorted_fits[1:], [np.inf]))
        m_down_sorted = np.concatenate(([np.inf], sorted_fits[:-1]))
        
        fit_range = np.max(sorted_fits) - np.min(sorted_fits)
        if fit_range == 0:
            distance_sorted = np.full_like(sorted_fits, np.inf)
        else:
            distance_sorted = (m_up_sorted - m_down_sorted) / fit_range

        ranks = np.zeros(num_individuals, dtype=int)
        ranks[idx_sorted_matlab] = np.arange(num_individuals)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            crowding += distance_sorted[ranks]
    
    crowding[np.isnan(crowding)] = np.inf

    del_idx = np.argsort(crowding, kind='mergesort')[:n_extra]

    REP['pos'] = np.delete(REP['pos'], del_idx, axis=0)
    REP['pos_fit'] = np.delete(REP['pos_fit'], del_idx, axis=0)

    REP = updateGrid(REP, ngrid)

    return REP


def LA_points(K, LB, UB, x0, scale_factor, d):
        
    K = np.array(K)
    LB = np.array(LB)
    UB = np.array(UB)
    x0 = np.array(x0)
            
    teta = np.random.rand()
            
    R = np.array([[np.cos(teta), -np.sin(teta)],
                    [np.sin(teta), np.cos(teta)]])
    K = K @ R
    
    dim_Xi = d + 2
    if dim_Xi % 2 != 0:
        dim_Xi += 1

    Xi = np.zeros((K.shape[0], dim_Xi))
            
    for i in range(0, d + 2, 2):
        gama = np.random.rand()            
                
        R = np.array([[np.cos(gama), -np.sin(gama)],
                        [np.sin(gama), np.cos(gama)]])
        Xi[:, i:i+2] = K @ R
    
    Xi = Xi[:, :d]

    scale = np.zeros(d)
    for i in range(d):
        scale[i] = scale_factor * (UB[i] - LB[i]) / (np.max(Xi[:, i]) - np.min(Xi[:, i]))
        Xi[:, i] = scale[i] * Xi[:, i]
    
    Pcc = np.zeros(d)
    delta = np.zeros(d)
    for i in range(d):
        Pcc[i] = (np.max(Xi[:, i]) - np.min(Xi[:, i])) / 2 + np.min(Xi[:, i])
        Xi[round(len(K) / 2) -1, i] = Pcc[i] 
        delta[i] = Pcc[i] - x0[i]
    
    X = np.zeros_like(Xi)
    for i in range(d):
        X[:, i] = Xi[:, i] - delta[i]
    
    return X
