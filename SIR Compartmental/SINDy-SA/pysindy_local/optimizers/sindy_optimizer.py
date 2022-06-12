import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from ..utils.base import supports_multiple_targets
from .base import _MultiTargetLinearRegressor


COEF_THRESHOLD = 1e-14


class SINDyOptimizer(BaseEstimator):
    """
    Wrapper class for optimizers/sparse regression methods passed into the SINDy object.

    Enables single target regressors (i.e. those whose predictions are 1-dimensional)
    to perform multi target regression (i.e. predictions are 2-dimensional).
    Also enhances an ``_unbias`` function to reduce bias when regularization is used.

    Parameters
    ----------
    optimizer: estimator object
        The optimizer/sparse regressor to be wrapped, implementing ``fit`` and
        ``predict``. ``optimizer`` should also have the attributes ``coef_``,
        ``fit_intercept``, ``normalize``, and ``intercept_``.

    unbias : boolean, optional (default True)
        Whether to perform an extra step of unregularized linear regression to unbias
        the coefficients for the identified support.
        For example, if ``optimizer=STLSQ(alpha=0.1)`` is used then the learned
        coefficients will be biased toward 0 due to the L2 regularization.
        Setting ``unbias=True`` will trigger an additional step wherein the nonzero
        coefficients learned by the optimizer object will be updated using an
        unregularized least-squares fit.
    """

    def __init__(self, optimizer, unbias=True, feature_names=None, symbolic_equations=None, data=None, time=None, get_feature_names=None):
        if not hasattr(optimizer, "fit") or not callable(getattr(optimizer, "fit")):
            raise AttributeError("optimizer does not have a callable fit method")
        if not hasattr(optimizer, "predict") or not callable(
            getattr(optimizer, "predict")
        ):
            raise AttributeError("optimizer does not have a callable predict method")

        self.optimizer = optimizer
        self.unbias = unbias
        self.optimizer.feature_names = feature_names
        self.optimizer.symbolic_equations = symbolic_equations
        self.optimizer.data = data
        self.optimizer.time = time
        self.optimizer.get_feature_names = get_feature_names

    def fit(self, x, y):
        if len(y.shape) > 1 and y.shape[1] > 1:
            if not supports_multiple_targets(self.optimizer):
                self.optimizer = _MultiTargetLinearRegressor(self.optimizer)
        self.optimizer.fit(x, y)
        if not hasattr(self.optimizer, "coef_"):
            raise AttributeError("optimizer has no attribute coef_")
        # self.ind_ = np.abs(self.coef_) > COEF_THRESHOLD

        if self.unbias:
            self._unbias(x, y)

        return self

    def _unbias(self, x, y):
        coef = np.zeros((y.shape[1], x.shape[1]))
        if hasattr(self.optimizer, "fit_intercept"):
            fit_intercept = self.optimizer.fit_intercept
        else:
            fit_intercept = False
        if hasattr(self.optimizer, "normalize"):
            normalize = self.optimizer.normalize
        else:
            normalize = False
        for i in range(self.ind_.shape[0]):
            if np.any(self.ind_[i]):
                coef[i, self.ind_[i]] = (
                    LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
                    .fit(x[:, self.ind_[i]], y[:, i])
                    .coef_
                )
        self.set_coefficients(coef)

    def predict(self, x):
        prediction = self.optimizer.predict(x)
        if prediction.ndim == 1:
            return prediction[:, np.newaxis]
        else:
            return prediction

    def set_coefficients(self, coef):
        if self.optimizer.coef_.ndim == 1:
            self.optimizer.coef_ = coef[0]
        else:
            self.optimizer.coef_ = coef

    @property
    def coef_(self):
        if self.optimizer.coef_.ndim == 1:
            return self.optimizer.coef_[np.newaxis, :]
        else:
            return self.optimizer.coef_

    @property
    def ind_(self):
        if self.optimizer.ind_.ndim == 1:
            return self.optimizer.ind_[np.newaxis, :]
        else:
            return self.optimizer.ind_

    @property
    def intercept_(self):
        if hasattr(self.optimizer, "intercept_"):
            return self.optimizer.intercept_
        else:
            return 0.0

    @property
    def X_dot_model_(self):
        return self.optimizer.X_dot_model_

    @property
    def SSE_(self):
        return self.optimizer.SSE_

    @property
    def SSE_data_(self):
        return self.optimizer.SSE_data_

    @property
    def relative_error_(self):
        return self.optimizer.relative_error_

    @property
    def Ftest_(self):
        return self.optimizer.Ftest_

    @property
    def Ftest_data_(self):
        return self.optimizer.Ftest_data_

    @property
    def mean_(self):
        return self.optimizer.mean_

    @property
    def epsilon_std_(self):
        return self.optimizer.epsilon_std_

    @property
    def num_eval_(self):
        return self.optimizer.num_eval_

    @property
    def mu_star_(self):
        return self.optimizer.mu_star_
    
    @property
    def sigma_(self):
        return self.optimizer.sigma_

    @property
    def param_min_(self):
        return self.optimizer.param_min_

    @property
    def history_(self):
        return self.optimizer.history_

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
