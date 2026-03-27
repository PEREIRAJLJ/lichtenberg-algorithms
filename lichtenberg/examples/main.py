from lichtenberg import mola
import matplotlib.pyplot as plt
import numpy as np

def objectives(x):    
    y1 = x[0]
    n = len(x)
    g = 1 + (9 / 29) * sum(x[1:n])
    h = 1 - np.sqrt(y1 / g) - (y1 / g) * np.sin(10 * np.pi * y1)
    y2 = g * h
    return np.array([y1, y2])

def constraints(x):    
    g = []
    geq = []
    return np.array(g), np.array(geq)

def main():
        
    # Optimizer Parameters
    LB = np.zeros(10)  # Lower bounds
    UB = np.ones(10)   # Upper bounds
    d = len(UB)        # Problem dimension
    pop = 10           # Population
    n_iter = 100       # Max number of iterations/generations
    ref = 0.4          # if more than zero, a second LF is created with refinement % the size of the other
    IntCon = 0         # Zero if there are no variables that must be integers
    ngrid = 30         # Number of grids in each dimension
    Nr = 100           # Maximum number of solutions in PF
    n_objectives = 2   # Number of objectives
        
    x, fval = mola.multi_objective_lichtenberg_algorithm(
        objectives, n_objectives, d, pop, LB, UB, ref, n_iter, ngrid, Nr, IntCon, constraints
    )
        
    plt.figure(figsize=(8, 5))
    plt.scatter(fval[:, 0], fval[:, 1], c='red', edgecolors='black', s=40)
    plt.title("MOLA", fontweight='bold')
    plt.xlabel("f₁")
    plt.ylabel("f₂")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
