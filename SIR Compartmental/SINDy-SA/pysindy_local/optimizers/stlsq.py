import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression
from sklearn.utils.validation import check_is_fitted

from .base import BaseOptimizer

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from SALib.sample import saltelli as saltelli_sample
from SALib.analyze import sobol as sobol_analyze
from scipy.integrate import solve_ivp


class STLSQ(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight array w that are below a given threshold.

    See the following reference for more details:

        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out.

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of sequentially thresholded least-squares.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = STLSQ(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        # threshold=0.1,
        alpha=0.05,
        # max_iter=20,
        ridge_kw=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        verbose=False,
        sa_method="Morris",
        bounds_perc=0.2,
        N=20,
        num_levels=4,
        # error_factor=100,
        window=3,
        epsilon=1.0,
        time=None,
        sa_times=None,
        non_physical_features=None,
    ):
        super(STLSQ, self).__init__(
            # max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        # if threshold < 0:
        #     raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        # self.threshold = threshold
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.verbose = verbose
        self.sa_method = sa_method
        self.bounds_perc = bounds_perc
        self.N = N
        self.num_levels = num_levels
        # self.error_factor = error_factor
        self.window = window
        self.epsilon = epsilon
        self.non_physical_features = non_physical_features

        if time is None or sa_times is None:
            self.sa_time_ind = [-1]
        else:
            self.sa_time_ind = []
            for t in sa_times:
                self.sa_time_ind.append(self._find_nearest(time, t))

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        return index

    # def _get_best_model(self, SSE, error_factor):
    #     best_model = len(SSE)
    #     for i in range(len(SSE)-2, -1, -1):
    #         flag = True
    #         for j in range(i+1, len(SSE)):
    #             if SSE[j] < error_factor*SSE[i]:
    #                 flag = False
    #                 break

    #         if flag:
    #             best_model = i+1
    #             break
    #     else:
    #         warnings.warn(
    #             "STLSQ._get_best_model cannot determine the best model using error_factor = {}.".format(
    #                 error_factor
    #             ),
    #             ConvergenceWarning,
    #         )

    #     return best_model

    def _error_has_increased(self, SSE, it, window, epsilon):
        if it == 0:
            return False

        if it == 1:
            mean = SSE[0]
            std = 0.1*SSE[0]
        else:
            mean = np.mean(SSE[:it]) if it < window else np.mean(SSE[(it - window):it])
            std = np.std(SSE[:it]) if it < window else np.std(SSE[(it - window):it])

        self.mean_.append(mean)
        self.epsilon_std_.append(epsilon*std)

        print_string = f"Comparing {str(mean)} + {str(epsilon)}*{str(std)} with {str(SSE[it])}: "
        if SSE[it] <= mean + epsilon*std:
            print_string += "error has NOT increased"
            if self.verbose:
                print(print_string)
            return False
        
        print_string += "error has increased"
        if self.verbose:
            print(print_string)
        return True

    def _SSE(self, target, predicted):
        squared_errors = (target - predicted)**2.0
        return np.sum(squared_errors)

    def _Ftest(self, loss, prev_coef, cur_coef, total_bases):
        prev_bases = np.count_nonzero(prev_coef != 0.0)
        cur_bases = np.count_nonzero(cur_coef != 0.0)

        Ftest = ((loss[-1] - loss[-2])/(prev_bases - cur_bases))/(loss[-2]/(total_bases - prev_bases))
        if self.verbose:
            print("Loss at the previous iteration = " + str(loss[-2]))
            print("Loss at the current iteration = " + str(loss[-1]))
            print("Number of bases at the previous iteration = " + str(prev_bases))
            print("Number of bases at the current iteration = " + str(cur_bases))
            print("Total number of bases = " + str(total_bases))
            print("F-test = " + str(Ftest))

        return Ftest

    def _get_less_important_param(self, ind, sensitivity_ind):
        total_points = np.zeros(ind.shape)
        for i in range(len(sensitivity_ind)):
            ST = np.zeros(ind.shape)
            ST[ind] = sensitivity_ind[i]
            for j in range(ST.shape[0]):
                sorted_ind = np.argsort(ST[j])
                total_points[j][sorted_ind[-1]] = -1

                points = 1
                for k in range(len(sorted_ind)-2, -1, -1):
                    if total_points[j][sorted_ind[k]] != -1:
                        total_points[j][sorted_ind[k]] += points
                        points += 1

        if self.verbose:
            print("Total points = " + str(total_points[ind]))

        if np.all(total_points[ind] == -1):
            return None
        return np.argwhere(total_points[ind] == np.amax(total_points[ind])).flatten()

    # def _get_less_important_param(self, n_params, sensitivity_ind):
    #     total_points = np.zeros(n_params)
    #     for i in range(len(sensitivity_ind)):
    #         sorted_ind = np.argsort(sensitivity_ind[i])
    #         total_points[sorted_ind[-1]] = -1

    #         points = 1
    #         for j in range(len(sorted_ind)-2, -1, -1):
    #             if total_points[sorted_ind[j]] != -1:
    #                 total_points[sorted_ind[j]] += points
    #                 points += 1

    #     if self.verbose:
    #         print("Total points = " + str(total_points))

    #     if np.all(total_points == -1):
    #         return -1
    #     return np.argmax(total_points)

    def _input_fmt(self, string):
        return string.replace(" ", "*").replace("^", "**").replace("cos", "np.cos").replace("sin", "np.sin").replace("log", "np.log")

    def _create_model_func(self, param_expression, var_expression, model_expression, n_vars):
        function = """def _model(t, X, """ + param_expression + """):
            import numpy as np
            """ + var_expression + """ = X
            dXdt = """ + (model_expression if n_vars == 1 else """[""" + model_expression + """]""") + """
            return dXdt
        """
        return function

    def _evaluate_model2(self, feature_library, param_value, rows, cols, ind):
        param_names = np.array(["c" + str(i) for i in range(len(ind.flatten()))])
        param_names = np.reshape(param_names, ind.shape)
        param_expression = ", ".join(param_names[ind].flatten()) + ","

        var_names = self.feature_names
        var_expression = ", ".join(var_names)

        self.ind_ = ind
        symbolic_equations = self.symbolic_equations(param_names, self._input_fmt)
        model_expression = ", ".join(symbolic_equations)

        wrapper = {}
        function = self._create_model_func(param_expression, var_expression, model_expression, len(var_names))
        exec(function, wrapper)

        x = solve_ivp(wrapper['_model'], [self.time[0], self.time[-1]], self.data[0, :], method='LSODA', t_eval=self.time, args=tuple(param_value))
        x = np.transpose(x.y)
        return x

    def _evaluate_model(self, feature_library, param_value, rows, cols, ind):
        p = np.zeros((rows, cols))
        p[ind] = param_value
        x_dot = np.matmul(feature_library, np.transpose(p))
        return x_dot

    def _sensitivity_analysis(self, feature_library, rows, cols, ind, coef, bounds_perc, N, num_levels, sa_time_ind):
        all_names = np.array(["c" + str(i) for i in range(len(ind.flatten()))])

        bounds = []
        for c in coef[ind].flatten():
            if c < 0.0:
                bounds.append([(1.0 + bounds_perc)*c, (1.0 - bounds_perc)*c])
            else:
                bounds.append([(1.0 - bounds_perc)*c, (1.0 + bounds_perc)*c])

        problem = {
            'num_vars': np.sum(ind),
            'names': all_names[ind.flatten()].tolist(),
            'bounds': bounds
        }

        if self.sa_method == "Morris":
            param_values = morris_sample.sample(problem, N, num_levels = num_levels)
        elif self.sa_method == "Sobol":
            param_values = saltelli_sample.sample(problem, N)
        self.num_eval_.append(param_values.shape[0])

        QoI = np.zeros((param_values.shape[0], feature_library.shape[0], rows))
        for i, param_value in enumerate(param_values):
            QoI[i] = self._evaluate_model(feature_library, param_value, rows, cols, ind)

        if self.sa_method == "Morris":
            ST_sum = []
            sa_times_mu_star = np.empty(len(sa_time_ind), dtype = object)
            sa_times_sigma = np.empty(len(sa_time_ind), dtype = object)
            for i, j in enumerate(sa_time_ind):
                Si = []
                ST = np.zeros((rows, np.sum(ind)))
                mu_star = np.zeros((rows, np.sum(ind)))
                sigma = np.zeros((rows, np.sum(ind)))
                for k in range(rows):
                    Si.append(morris_analyze.analyze(problem, param_values, QoI[:, j, k], num_levels = num_levels))
                    ST[k] = np.sqrt(Si[k]['mu_star']**2.0 + Si[k]['sigma']**2.0)
                    mu_star[k] = Si[k]['mu_star']
                    sigma[k] = Si[k]['sigma']
                ST_sum.append(ST.sum(axis = 0))
                sa_times_mu_star[i] = mu_star.sum(axis = 0)
                sa_times_sigma[i] = sigma.sum(axis = 0)
            self.mu_star_.append(sa_times_mu_star)
            self.sigma_.append(sa_times_sigma)
        elif self.sa_method == "Sobol":
            ST_sum = []
            for j in sa_time_ind:
                Si = []
                ST = np.zeros((rows, np.sum(ind)))
                for k in range(rows):
                    Si.append(sobol_analyze.analyze(problem, QoI[:, j, k]))
                    ST[k] = Si[k]['ST']
                ST_sum.append(ST.sum(axis = 0))

        if self.verbose:
            print("ST_sum = " + str(ST_sum))

        param_min = self._get_less_important_param(ind, ST_sum)
        if param_min is None:
            return ind.flatten()
        self.param_min_.append(param_min)

        if self.verbose:
            print("Less important parameter = " + str(param_min))

        big_ind = ind.flatten()
        nonzero = -1
        for i in range(len(big_ind)):
            if big_ind[i]:
                nonzero += 1
            if nonzero in param_min:
                big_ind[i] = False

        return big_ind

    def _sparse_coefficients(self, feature_library, rows, cols, ind, coef, bounds_perc, N, num_levels, sa_time_ind):
        """Perform thresholding of the weight vector(s)"""
        c = coef.flatten()
        # big_ind = np.abs(c) >= threshold
        big_ind = self._sensitivity_analysis(feature_library, rows, cols, ind, coef, bounds_perc, N, num_levels, sa_time_ind)
        c[~big_ind] = 0
        return c.reshape((rows, cols)), big_ind.reshape((rows, cols))

    def _regress(self, x, y):
        """Perform the ridge regression"""
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw, solver = 'auto')
        self.iters += 1
        return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        ind = self.ind_
        n_non_physical = 0
        if self.non_physical_features is not None:
            feature_names = self.get_feature_names()
            for i in range(ind.shape[0]):
                for j, feature in enumerate(feature_names):
                    if feature in self.non_physical_features[i]:
                        ind[i, j] = False
                        n_non_physical += 1

        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)

        self.X_dot_model_ = []
        self.SSE_ = []
        # self.SSE_data_ = []
        self.relative_error_ = []
        self.Ftest_ = []
        # self.Ftest_data_ = []
        self.mean_ = []
        self.epsilon_std_ = []
        self.num_eval_ = []
        self.mu_star_ = []
        self.sigma_ = []
        self.param_min_ = []

        self.max_iter = n_targets*n_features
        for it in range(self.max_iter):
            if self.verbose:
                print("---------- ITERATION " + str(it+1) + " ----------")

            # if np.count_nonzero(ind) == 0:
            #     warnings.warn(
            #         "Sparsity parameter is too big ({}) and eliminated all "
            #         "coefficients".format(self.threshold)
            #     )
            #     coef = np.zeros((n_targets, n_features))
            #     break

            coef = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    # warnings.warn(
                    #     "Sparsity parameter is too big ({}) and eliminated all "
                    #     "coefficients".format(self.threshold)
                    # )
                    continue
                coef_i = self._regress(x[:, ind[i]], y[:, i])
                coef[i, ind[i]] = coef_i

            QoI = self._evaluate_model(
                x, coef[ind].flatten(), n_targets, n_features, ind
            )
            QoI2 = self._evaluate_model2(
                x, coef[ind].flatten(), n_targets, n_features, ind
            )
            self.X_dot_model_.append(QoI)
            self.SSE_.append(self._SSE(y, QoI))
            # self.SSE_data_.append(self._SSE(self.data, QoI2))
            if it > 0:
                self.relative_error_.append((self.SSE_[-1] - self.SSE_[-2])/self.SSE_[-2])
            if it > 1:
                self.Ftest_.append(self._Ftest(self.SSE_, self.history_[-1], coef, n_targets*n_features - n_non_physical))
                # self.Ftest_data_.append(self._Ftest(self.SSE_data_, self.history_[-1], coef, n_targets*n_features - n_non_physical))

            if self.verbose:
                print("SSE = " + str(self.SSE_[-1]))

            if self._error_has_increased(self.SSE_, it, self.window, self.epsilon):
                break

            self.history_.append(coef)

            if self.verbose:
                print("Coefficients before sensitivity analysis = " + str(coef))
            sparse_coef, sparse_ind = self._sparse_coefficients(
                x, n_targets, n_features, ind, coef, self.bounds_perc, self.N, self.num_levels, self.sa_time_ind
            )
            if self.verbose:
                print("Coefficients after sensitivity analysis = " + str(sparse_coef) + "\n")

            if np.sum(ind) == np.sum(sparse_ind):
                # could not (further) select important features
                break

            coef = sparse_coef
            ind = sparse_ind

            # if np.sum(ind) == n_features_selected or self._no_change():
            #     # could not (further) select important features
            #     break
        # else:
        #     warnings.warn(
        #         "STLSQ._reduce did not converge after {} iterations.".format(
        #             self.max_iter
        #         ),
        #         ConvergenceWarning,
        #     )
        #     try:
        #         coef
        #     except NameError:
        #         coef = self.coef_
        #         warnings.warn(
        #             "STLSQ._reduce has no iterations left to determine coef",
        #             ConvergenceWarning,
        #         )

        self.coef_ = self.history_[-1]
        self.ind_ = self.coef_ != 0.0

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )
