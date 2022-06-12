import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylops
import os

class DataDenoising:
	def __init__(self, X = None, t = None, feature_names = None):
		self.X = X
		self.t = t
		self.feature_names = feature_names

		if self.X is not None and self.t is not None:
			data = np.zeros((X.shape[0], X.shape[1]+1))
			data[:,0] = self.t
			data[:,1:] = self.X

			columns = ["t"] + feature_names

			self.dataset = pd.DataFrame(
				data = data,
				columns = columns
			)
	
	def simple_moving_average(self, window = 3):
		dataset_copy = self.dataset.iloc[:,1:].copy()
		for column in dataset_copy:
			dataset_copy[column] = dataset_copy[column].rolling(window, min_periods = 1).mean()

		return dataset_copy.to_numpy()

	def exponential_moving_average(self, alpha = 0.1, adjust = False):
		dataset_copy = self.dataset.iloc[:,1:].copy()
		for column in dataset_copy:
			dataset_copy[column] = dataset_copy[column].ewm(alpha = alpha, adjust = adjust).mean()

		return dataset_copy.to_numpy()

	def l2_regularization(self, lambda_ = 1.0e2):
		Iop = pylops.Identity(self.t.shape[0])
		D2op = pylops.SecondDerivative(self.t.shape[0], edge = True)

		X_l2r = np.zeros(self.X.shape)
		for j in range(X_l2r.shape[1]):
			Y = Iop*self.X[:,j]
			X_l2r[:,j] = pylops.optimization.leastsquares.RegularizedInversion(Iop, [D2op], Y, 
				epsRs = [np.sqrt(lambda_/2.0)], 
				**dict(iter_lim = 30)
			)

		return X_l2r

	def total_variation_regularization(self, mu = 0.01, lambda_ = 0.3, niter_out = 50, niter_in = 3):
		Iop = pylops.Identity(self.t.shape[0])
		Dop = pylops.FirstDerivative(self.t.shape[0], edge = True, kind = 'backward')

		X_tvr = np.zeros(self.X.shape)
		for j in range(X_tvr.shape[1]):
			Y = Iop*self.X[:,j]
			X_tvr[:,j], niter = pylops.optimization.sparsity.SplitBregman(Iop, [Dop], Y, 
				niter_out, niter_in, mu = mu, epsRL1s = [lambda_], 
				tol = 1.0e-4, tau = 1.0, 
				**dict(iter_lim = 30, damp = 1.0e-10)
			)

		return X_tvr

	def plot_derivative(self, X_dot = None, t = None, init_cond_id = None, X0 = None):
		if X_dot is not None and t is not None:
			markers = ["o", "^", "s", "p", "P", "*", "X", "d"]

			fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
			for i, feature_name in enumerate(self.feature_names):
				ax.plot(t, X_dot[:,i], "k" + markers[i], label = r"$" + feature_name + "'(t)$", alpha = 0.5, markersize = 3)
			for i, feature_name in enumerate(self.feature_names):
				ax.plot(t, X_dot[:,i], label = r"$" + feature_name + "'(t)$", alpha = 1.0, linewidth = 1)
			ax.set(xlabel = r"Time $t$", ylabel = r"$X'(t)$",
				# title = "Derivative - Initial condition = " + str(X0)
			)
			ax.legend()
			# fig.show()
			plt.savefig(os.path.join("output", "deriv_ic" + str(init_cond_id) + ".png"), bbox_inches = 'tight')
			plt.close()

	def plot_sma(self, windows = None):
		if windows is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for window in windows:
					X_sma = self.simple_moving_average(window)
					ax.plot(self.t, X_sma[:,i], label = "SMA(" + str(window) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Simple Moving Average - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "SMA_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_ema(self, alphas = None, adjusts = None):
		if alphas is not None and adjusts is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for alpha in alphas:
					for adjust in adjusts:
						X_ema = self.exponential_moving_average(alpha, adjust)
						ax.plot(self.t, X_ema[:,i], label = "EMA(" + str(alpha) + ", " + str(adjust) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Exponential Moving Average - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "EMA_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_l2r(self, lambdas = None):
		if lambdas is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for lambda_ in lambdas:
					X_l2r = self.l2_regularization(lambda_ = lambda_)
					ax.plot(self.t, X_l2r[:,i], label = "L2R(" + str(lambda_) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"L2 Regularization - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "L2R_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_tvr(self, mus = None, lambdas = None):
		if mus is not None and lambdas is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for mu in mus:
					for lambda_ in lambdas:
						X_tvr = self.total_variation_regularization(mu = mu, lambda_ = lambda_)
						ax.plot(self.t, X_tvr[:,i], label = "TVR(" + str(mu) + ", " + str(lambda_) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Total Variation Regularization - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "TVR_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()