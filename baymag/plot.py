import matplotlib.pyplot as plt


def predictplot(prediction, ylabel=None, x=None, xlabel=None, ax=None):
    """Lineplot of prediction with uncertainty estimate

    Parameters
    ----------
    prediction : bayspar.predict.Prediction
        MCMC prediction
    ylabel : string, optional
        String label for y-axis.
    x : numpy.ndarray, optional
        Array over which to evaluate the densities. Default is
        `numpy.arange(0, 40.1, 0.1)`.
    xlabel : string, optional
        String label for x-axis.
    ax : matplotlib.Axes, optional
        Axes to plot onto.

    Returns
    -------
    ax : matplotlib.Axes
    """
    if ax is None:
        ax = plt.gca()

    if x is None:
        x = list(range(len(prediction.ensemble)))

    perc = prediction.percentile(q=[5, 50, 95])

    ax.fill_between(x, perc[:, 0], perc[:, 2], alpha=0.25,
                    label='90% uncertainty', color='C0')

    ax.plot(x, perc[:, 1], label='Median', color='C0')
    ax.plot(x, perc[:, 1], marker='.', color='C0')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax


def densityplot(prediction, ax=None):
    raise NotImplementedError
