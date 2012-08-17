import numpy as np
import regreg.api as rr

import statsmodels.discrete.discrete_model as dm

n, p = 100, 500
X = np.random.standard_normal((n, p))
beta = np.zeros(p)
beta[:3] = np.array([1,4,3.]) / 10.

Y = np.random.poisson(np.exp(np.dot(X,beta))) * 1.
model = dm.Poisson(Y,X)

class statsmodel(rr.smooth_atom):

    def __init__(self, model, quadratic=None,initial=None,coef=1.):
        shape = model.exog.shape[1:]
        rr.smooth_atom.__init__(self, shape, 
                                offset=None,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

    def smooth_objective(self, x, mode='func', check_feasibility=False):
        if mode == 'func':
            return -model.loglike(x)
        elif mode == 'grad':
            return -model.score(x)
        elif mode == 'both':
            return -model.loglike(x), -model.score(x)
        else:
            raise ValueError('mode incorrectly specified')

def lasso_example():

    penalty = rr.l1norm(p, lagrange=2.)
    problem = rr.simple_problem(statsmodel(model), penalty)
    soln = problem.solve()

def lasso_bound_example():

    penalty = rr.l1norm(p, bound=1.)
    problem = rr.simple_problem(statsmodel(model), penalty)
    soln = problem.solve()

def group_lasso_example():
    # group lasso penalty

    penalty_structure = np.zeros(p)
    
    # have first 5 unpenalized
    
    penalty_structure[:5] = rr.UNPENALIZED

    # the next 15 have them be penalized with a LASSO and nonnegatively constrained

    penalty_structure[5:10] = rr.POSITIVE_PART

    # from 10 to 20 use the LASSO

    penalty_structure[10:20] = rr.L1_PENALTY
    
    # the rest, form groups of size 20
    # group labels should not be in [rr.UNPENALIZED, rr.L1_PENALTY, rr.POSITIVE_PART] == [-3,-2,-1]
    for idx, j in enumerate(range(1,25)):
        penalty_structure[j*(20):(j+1)*20] = idx

    penalty = rr.group_lasso(penalty_structure, 0.3)
    problem = rr.simple_problem(statsmodel(model), penalty)
    soln = problem.solve()

if __name__ == "__main__":
    lasso_example()
    lasso_bound_example()
    group_lasso_example()
