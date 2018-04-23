import numpy as np

def plot_res(y_target, y_hat, title=None, env='jupyter', engine='matplotlib'):
    
    if engine=='matplotlib':
        _plot_res_matplotlib(y_target, y_hat, title=title)
    elif engine=='bokeh':
        _plot_res_bokeh(y_target, y_hat, title=title, env=env)
    else:
        raise ValueError("Unknown engine: '%s'..."%engine)
    return None

def _plot_res_matplotlib(y_target, y_hat, title=None):
    import matplotlib
    import matplotlib.pyplot as plt
    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(8,3.5), dpi=100)
    ax.plot(range(len(y_target)), y_target, label='Q_obs')
    ax.plot(range(len(y_hat)), y_hat, label='Q_sim', c='orangered')

    ax.set(xlabel='Time (hours)', ylabel='Discharge (m³/s)',
           title=title)
    ax.grid()

    # fig.savefig("res.png")
    plt.show()
    return None

def _plot_res_bokeh(y_target, y_hat, title=None, env='jupyter'):
    import bokeh.plotting as plt
    from bokeh.layouts import column, row
    from bokeh.io import output_notebook
    from bokeh.models import LinearAxis, Range1d, Title, Legend

    if env=='jupyter':
        notebook_handle = True
        output_notebook()
    
    # plot 1 with numerical x index (Time hours) 
    p1 = plt.figure(plot_width=800, plot_height=350)
    
    plt.hold()
    # Title
    titre = Title(text=title, align='center', text_font_size='12pt')
    
    # Axis
    p1.yaxis.axis_label = "Discharge (m³/s)"
    p1.xaxis.axis_label = "Time (hours)"

    # plot discharge
    p1.line(x=range(len(y_hat)), y=y_target, line_width=2.0, legend='Q_obs')
    p1.line(x=range(len(y_hat)), y=y_hat, line_width=2.0, color='orangered', legend='Q_sim')
    
    p1.add_layout(titre, 'above')
    
    t = plt.show(p1, notebook_handle=notebook_handle)
    return None

def plot_err(err_tr, err_cv, title=None,
             xlim=(None, None), ylim=(None, None),
             env='jupyter', engine='matplotlib'):
    if engine=='matplotlib':
        _plot_err_matplotlib(err_tr, err_cv, title=title, xlim=xlim, ylim=ylim)
    elif engine=='bokeh':
        _plot_err_bokeh(err_tr, err_cv, title=title, env=env, xlim=xlim, ylim=ylim)
    else:
        raise ValueError("Unknown engine: '%s'..."%engine)
    return None

def _plot_err_matplotlib(err_tr, err_cv, title=None,
             xlim=(None, None), ylim=(None, None)):
    import matplotlib
    import matplotlib.pyplot as plt
    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    ax.plot(range(len(err_tr)), err_tr, label='Training Loss')
    ax.plot(range(len(err_cv)), err_cv, 
        label='Cross-validation MSE', c='orangered')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    ax.set(xlabel='Iteration [N]', ylabel='Loss',
           title=title)
    ax.grid()

    # fig.savefig("res.png")
    plt.show()
    return None

def _plot_err_bokeh():
    pass

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate a simple plot of the test and training learning curve.

#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.

#     title : string
#         Title for the chart.

#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.

#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.

#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.

#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - An object to be used as a cross-validation generator.
#           - An iterable yielding train/test splits.

#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.

#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     matplt.figure()
#     matplt.title(title)
#     if ylim is not None:
#         matplt.ylim(*ylim)
#     matplt.xlabel("Training examples")
#     matplt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     matplt.grid()

#     matplt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     matplt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     matplt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     matplt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     matplt.legend(loc="best")
#     return matplt
