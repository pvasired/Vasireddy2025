# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
import sklearn.model_selection as model_selection
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
import copy

def negLL_hotspot(params, *args):
    """
    Compute the negative log likelihood for a logistic regression
    binary classification task assuming the hotpot model of activation.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X, d+1 (see below)
    *args (tuple): X (np.ndarray) stimulation amplitudes with column of 1s, shape N x (d+1),
                   y (np.ndarray) response probabilities, shape N x 1,
                   trials (np.ndarray) number of trials at each amplitude, shape N x 1,
                   verbose (bool) increases verbosity
                   method (str): regularization method. 'l1', 'l2', and
                                 'MAP' with multivariate Gaussian prior 
                                 are supported.
                   reg (float): regularization parameter
                                In the case of MAP, reg consists of 
                                (regmap, mu, cov)
                                where regmap is a constant scalar
                                      mu is the mean vector
                                      cov is the covariance matrix

    Returns:
    negLL (float): negative log likelihood of the data given the 
                   current parameters, possibly plus a regularization
                   term.
    """
    X, y, trials, verbose, method, reg = args
    
    w = params.reshape(-1, X.shape[-1]).astype(float)

    # Get predicted probability of spike using current parameters
    response_mat = expit(X @ w.T)

    episilon = 1e-9
    yPred = np.clip(1 - np.multiply.reduce(1 - response_mat, axis=1), episilon, 1 - episilon)
    
    # negative log likelihood for logistic
    NLL = -np.sum(trials * (y * np.log(yPred) + (1 - y) * np.log(1 - yPred)))
    ###

    # Add the regularization penalty term if desired
    penalty = 0
    if reg > 0:
        if method == 'l1':
            # penalty term according to l1 regularization
            penalty = reg*np.linalg.norm(w.flatten(), ord=1)
        elif method == 'l2':
            # penalty term according to l2 regularization
            penalty = reg/2*np.linalg.norm(w.flatten())**2
        elif method == 'MAP':
            regmap, mu, cov = reg
            # penalty term according to MAP with Gaussian prior
            penalty = regmap * 0.5 * (params - mu) @ np.linalg.inv(cov) @ (params - mu)

    if verbose:
        print(NLL, penalty)
        
    return(NLL + penalty)

# Numpy version of activation_probs()
def sigmoidND_nonlinear(X, w):
    """
    N-dimensional nonlinear sigmoid computed according to multi-
    hotspot model.
    
    Parameters:
    X (np.ndarray): Input amplitudes N x (d+1)
    w (np.ndarray): Weight vector matrix m x (d+1) where m is the number of hotspots

    Returns:
    response (np.ndarray): Probabilities with same length as X
    """
    response_mat = expit(X @ w.T)
    response = 1 - np.multiply.reduce(1 - response_mat, axis=1)
    return response
    
def fit_surface_earlystop(X_expt, probs, T, w_inits_, 
                          R2_thresh=0.1, test_size=0.2,
                          reg_method='l2', reg=0, 
                          slope_bound=100, zero_prob=0.01,
                          opt_verbose=False, verbose=True,
                          method='L-BFGS-B', random_state=None):
    """
    Fitting function for fitting surfaces to nonlinear data with multi-hotspot model.
    This function is primarily a wrapper for calling get_w() in the framework of 
    early stopping using the McFadden pseudo-R2 metric.

    Parameters:
    X_expt (N x d np.ndarray): Input amplitudes
    probs (N x 1 np.ndarray): Probabilities corresponding to the amplitudes
    T (N x 1 np.ndarray): Trials at each amplitude
    w_inits_ (list): List of initial guessses for each number of hotspots. Each element
                    in the list is a (m x (d + 1)) np.ndarray with m the number of 
                    hotspots. This list should be generated externally.
    R2_thresh (float): Threshold used for determining when to stop adding hotspots
    test_size (float): Size of the test set for early stopping
    reg_method (string): Regularization method. 'l2', 'l1', and 'MAP' with multivariate
                         Gaussian prior are supported.
    reg (float): regularization parameter
                                In the case of MAP, reg consists of 
                                (regmap, mu, cov)
                                where regmap is a constant scalar
                                      mu is the mean vector
                                      cov is the covariance matrix
    slope_bound (float): Bound on the slope of the weights
    zero_prob (float): Value for what the probability should be forced to be below
                       at an amplitude of 0-vector
    opt_verbose (bool): Increases verbosity of optimization
    verbose (bool): Increases verbosity
    method (string): Method for optimization according to constrained optimization
                     methods available in scipy.optimize.minimize
    random_state (int): Random seed for train/test split

    Returns:
    opt: A tuple consisting of 
         (0) The optimized set of parameters for the 
                optimized number of hotspots m using
                McFadden Pseudo-R2 and early stopping
         (1) The value of negLL_hotspot for the optimized parameters
         (2) The McFadden Pseudo-R2 for the optimized parameters
    w_inits (list): The new initial guesses for each number of hotspots for the
                    next possible iteration of fitting
    """
    w_inits = copy.deepcopy(w_inits_)

    # Return degenerate parameters if there are no data points
    if len(probs) == 0 or np.amax(probs) == 0:
        deg_opt = np.zeros_like(w_inits[-1])
        deg_opt[:, 0] = np.ones(len(deg_opt)) * -np.inf

        return (deg_opt, 0, -1), w_inits

    X_const = sm.add_constant(X_expt, has_constant='add')
    X_train, X_test, y_train, y_test, T_train, T_test = model_selection.train_test_split(X_const, probs, T,
                                                                                         test_size=test_size, random_state=random_state)

    test_R2s = np.zeros(len(w_inits))
    opts = []
    for i in range(len(w_inits)):
        if reg_method == 'MAP':
            opt = get_w(w_inits[i], X_train, y_train, T_train, zero_prob=zero_prob,
                                        method=method, 
                                        reg_method=reg_method,
                                        reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                                        verbose=opt_verbose, 
                                        slope_bound=slope_bound)
        else:
            opt = get_w(w_inits[i], X_train, y_train, T_train,
                                                        zero_prob=zero_prob, 
                                                        method=method, 
                                                        reg_method=reg_method, 
                                                        reg=reg, 
                                                        verbose=opt_verbose,
                                                        slope_bound=slope_bound)
        test_fun = negLL_hotspot(opt[0], X_test, y_test, T_test, opt_verbose, reg_method, reg[0])

        # Compute the negative log likelihood of the null model which only
        # includes an intercept
        ybar_test = np.mean(y_test)
        beta_null_test = np.log(ybar_test / (1 - ybar_test))
        null_weights_test = np.concatenate((np.array([beta_null_test]), 
                                             np.zeros(X_expt.shape[-1])))
        nll_null_test = negLL_hotspot(null_weights_test, X_test, y_test, T_test, False, reg_method, reg[0])

        test_R2 = 1 - test_fun / nll_null_test
        if verbose:
            print(f'Number of sites: {len(w_inits[i])}, Test R2: {test_R2}')
        test_R2s[i] = test_R2
        opts.append(opt)
        w_inits[i] = opt[0]

        if i > 0:
            if test_R2s[i-1] > 0 and (test_R2s[i] - test_R2s[i-1]) / test_R2s[i-1] <= R2_thresh:
                return opts[i-1], w_inits
            
    return opt, w_inits
    
def get_w(w_init, X, y, T, zero_prob=0.01, method='L-BFGS-B', 
          reg_method='l2', reg=0, slope_bound=100, bias_bound=None, verbose=False,
          options={'maxiter': 200000, 'ftol': 1e-15, 'maxfun': 200000}):
    """
    Fitting function for fitting data with a specified number of hotspots
    
    Parameters:
    w_init (m x (d + 1) np.ndarray): Initial guesses on parameters for model
                                     with m hotspots
    X (N x (d + 1) np.ndarray): Input amplitudes
    y (N x 1 np.ndarray): Probabilities corresponding to the amplitudes
    T (N x 1 np.ndarray): Trials at each amplitude
    zero_prob (float): The forced maximum probability at 0-vector
    method (string): Optimization method according to constrained optimization
                     methods available in scipy.optimize.minimize
    reg_method (string): Regularization method, 'l2', 'l1', and 'MAP' with multivariate
                            Gaussian prior are supported.
    reg (float): regularization parameter
                            In the case of MAP, reg consists of 
                            (regmap, mu, cov)
                            where regmap is a constant scalar
                                  mu is the mean vector
                                  cov is the covariance matrix
    slope_bound (float): Bound on the slope of the weights
    bias_bound (float): Bound on the bias of the weights
    verbose (bool): Increases verbosity
    options (dict): Options for optimization

    Returns:
    A tuple consisting of:
        (0) weights (m x (d + 1) np.ndarrray): Fitted weight vector
        (1) opt.fun (float): Minimized value of negative log likelihood
        (2) R2 (float): McFadden pseudo-R2 value
    """

    z = 1 - (1 - zero_prob)**(1/len(w_init))

    # Set up bounds for constrained optimization
    bounds = []
    for j in range(len(w_init)):
        bounds += [(bias_bound, np.log(z/(1-z)))]
        for i in range(X.shape[-1] - 1):
            bounds += [(-slope_bound, slope_bound)]

    ybar = np.mean(y)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), 
                                   np.zeros(X.shape[-1]-1)))
    nll_null = negLL_hotspot(null_weights, X, y, T, False, reg_method, reg)

    # Optimize the weight vector with MLE
    opt = minimize(negLL_hotspot, x0=w_init.ravel(), bounds=bounds,
                       args=(X, y, T, verbose, reg_method, reg), method=method,
                        options=options)
    
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)