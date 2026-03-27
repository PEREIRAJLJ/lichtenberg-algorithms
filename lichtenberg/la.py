# ============================================================
# LICHTENBERG ALGORITHM (LA) — PYTHON VERSION
# ============================================================
import os
import numpy as np
from scipy.io import loadmat
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# USER OBJECTIVE & CONSTRAINTS
# ============================================================

def objective(x):
    return np.sum(x**2)

def constraint(x):
    g = []    # inequality
    geq = []  # equality
    return g, geq

# ============================================================
# CONSTRAINT HANDLING (PENALTY)
# ============================================================

PEN = 1e15

def get_constraints(x):
    g, geq = constraint(x)
    penalty = 0.0
    for gi in g:
        if gi > 0:
            penalty += PEN * gi**2
    for gj in geq:
        if gj != 0:
            penalty += PEN * gj**2
    return penalty

def fitness(x):
    return objective(x) + get_constraints(x)

# ============================================================
# LOAD PRECOMPUTED LICHTENBERG FIGURE
# ============================================================

def load_LF(d):
    if d == 3:
        path = os.path.join(BASE_DIR, "LF3D.mat")
        data = loadmat(path)
        LF = data["LF3D"]
    else:
        path = os.path.join(BASE_DIR, "LFND.mat")
        data = loadmat(path)
        LF = data["LFND"]
    return LF

# ============================================================
# LA POINT GENERATION
# ============================================================

def LA_points(K, LB, UB, x0, scale_factor):
    K = K.astype(float)
    d = len(LB)

    # random rotation (2D only, as in MATLAB)
    if d != 3:
        theta = np.random.rand()
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        K[:, :2] = K[:, :2] @ R.T

    Xi = K[:, :d].copy()

    # scale to search space
    for i in range(d):
        denom = np.max(Xi[:, i]) - np.min(Xi[:, i])
        if denom == 0:
            denom = 1
        scale = scale_factor * (UB[i] - LB[i]) / denom
        Xi[:, i] *= scale

    # center around best
    delta = np.mean(Xi, axis=0) - x0
    X = Xi - delta

    return X

# ============================================================
# BOUND CHECK
# ============================================================

def bound_check(x, LB, UB):
    return np.minimum(np.maximum(x, LB), UB)

# ============================================================
# LICHTENBERG ALGORITHM CORE
# ============================================================

def LA_optimization(LB, UB, pop, n_iter, ref=0.4):
    d = len(LB)
    LB = np.array(LB)
    UB = np.array(UB)

    # Initial population
    Individuals = LB + (UB - LB) * np.random.rand(pop, d)
    Fitness = np.array([fitness(ind) for ind in Individuals])

    best_idx = np.argmin(Fitness)
    best = Individuals[best_idx].copy()
    fmin = Fitness[best_idx]

    # Load LF (precomputed)
    LF = load_LF(d)

    for t in range(n_iter):
        scale_factor = 1.2 * np.random.rand()

        X = LA_points(LF, LB, UB, best, scale_factor)

        if ref > 0:
            X_local = LA_points(LF, LB * ref, UB * ref, best, scale_factor)

        for i in range(pop):
            if ref > 0:
                n_local = int(0.4 * pop)
                n_global = pop - n_local
                S = np.vstack((
                    X[np.random.choice(len(X), n_global, replace=False)],
                    X_local[np.random.choice(len(X_local), n_local, replace=False)]
                ))
            else:
                S = X[np.random.choice(len(X), pop, replace=False)]

            candidate = bound_check(S[i], LB, UB)
            Fnew = fitness(candidate)

            if Fnew <= Fitness[i]:
                Individuals[i] = candidate
                Fitness[i] = Fnew

            if Fnew <= fmin:
                best = candidate.copy()
                fmin = Fnew

        if (t + 1) % 10 == 0:
            print(f"Iter {t+1:4d} | Best Fitness = {fmin:.6e}")

    return best, fmin

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    LB = [-5, -5]
    UB = [5, 5]

    pop = 200        # você pode subir depois
    n_iter = 100

    best, fmin = LA_optimization(
        LB=LB,
        UB=UB,
        pop=pop,
        n_iter=n_iter,
        ref=0.4
    )

    print("\n=== FINAL RESULT ===")
    print("Best solution:", best)
    print("Best fitness :", fmin)
