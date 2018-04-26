import numpy as np 

def _nse(q_rec, q_sim):
    '''
    ===
    NSE
    ===

    Nash-Sutcliffe efficiency. Metric for the estimation of performance of the 
    hydrological model

    Parameters
    ----------
    q_rec : array_like [n]
    Measured discharge [m3/s]
    q_sim : array_like [n] 
    Simulated discharge [m3/s]

    Returns
    -------
    f : float
    NSE value
    '''
    m = np.nanmean(q_rec)
    a = np.square(np.subtract(q_rec, q_sim))
    b = np.square(np.subtract(q_rec, m))
    if a.any() < 0.0:
        return(np.nan)
    f = 1.0 - (np.nansum(a)/np.nansum(b))
    return f

def _rmse(q_rec, q_sim):
    '''
    ====
    RMSE
    ====

    Sign-fliped Root Mean Squared Error. Metric for the estimation of performance of the 
    hydrological model.

    Parameters
    ----------
    q_rec : array_like [n]
    Measured discharge [m3/s]
    q_sim : array_like [n] 
    Simulated discharge [m3/s]

    Returns
    -------
    f : float
    RMSE value
    '''
    erro = np.square(np.subtract(q_rec,q_sim))
    if erro.any() < 0:
        return(np.nan)
    f = np.sqrt(np.nanmean(erro))
    return f

def _mse(y, y_pred):
    '''
    ===
    MSE
    ===

    Sign-fliped Mean Squared Error. Metric for the estimation of performance of the 
    hydrological model.

    Parameters
    ----------
    q_rec : array_like [n]
    Measured discharge [m3/s]
    q_sim : array_like [n] 
    Simulated discharge [m3/s]

    Returns
    -------
    f : float
    MSE value
    '''
    erro = np.square(np.subtract(y,y_pred))
    if erro.any() < 0:
        return(np.nan)
    f = np.nanmean(erro)
    return f

def _kge(q_rec, q_sim):
    '''
    ====
    KGE
    ====

    Kling-Gupta Efficiency. Non-dimensional perfomance estimator for hydrological models.

    Parameters
    ----------
    q_rec : array_like [n]
    Measured discharge [m3/s]
    q_sim : array_like [n] 
    Simulated discharge [m3/s]
    r     : float
    Lineary correlation coefficient of Q_sim and Q_obs
    alpha : float
    Relative variability in the simulated and observed values
    beta  : float
    

    Returns
    -------
    f : float
    KGE value
    '''
    r = np.corrcoef(q_rec, q_sim)[0][1]
    alpha = np.std(q_sim, axis=0)/np.std(q_rec, axis=0)
    beta = np.nanmean(q_sim)/np.nanmean(q_rec)
    f = 1 - np.sqrt((r-1.0)**2.0 + (alpha-1.0)**2.0 + (beta-1.0)**2.0)
    return f