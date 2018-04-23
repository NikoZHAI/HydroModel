from ._utils import _nse, _rmse, _kge
import numpy as np
from sklearn.metrics import make_scorer

def render_score(score_name, maximize=None):
    '''
    ====
    hydro_scorer.render_score
    ====

    Render a hydrological-used scorring function. 
    Including Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE) and Root-Mean-Squared-Error (RMSE).

    Parameters
    ----------
    score_name : string
    Name of the desired scoring function

    maximize : Boolean or None 
    Whether to maximize score

    Returns
    -------
    scorer : callable
    sklearn.metrics.scorer
    '''

    fun = None
    if score_name in ['KGE', 'NSE', 'RMSE']:
        if score_name == 'NSE':
            fun = _nse
            default_policy = True
        elif score_name == 'KGE':
            fun = _kge
            default_policy = True
        else:
            fun = _rmse
            default_policy = False
    else:
        raise 'ScoreNameError: No hydro score named ' + score_name

    greater_is_better = default_policy if maximize == None else maximize

    return make_scorer(fun, greater_is_better=greater_is_better)

