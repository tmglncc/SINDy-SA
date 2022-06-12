import numpy as np
import matplotlib.pyplot as plt
import os

class ModelPlots:
	def __init__(self, model = None, optimizer_method = "STLSQ+SA", experiment_id = 0):
		self.model = model
		self.optimizer_method = optimizer_method
		self.experiment_id = experiment_id

	def plot_sse(self):
		model_sse = self.model.get_SSE()
		model_mean = self.model.get_mean()
		model_epsilon_std = self.model.get_epsilon_std()
		max_iter = len(model_sse)
		model_history = self.model.get_history()

		# n_terms = np.zeros(len(model_history)-1)
		# for it in range(1, len(model_history)):
		# 	n_terms[it-1] = np.count_nonzero(model_history[it] != 0.0)

		fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
		rects = ax.bar(range(max_iter), model_sse, width = 0.25, color = "b", alpha = 0.5, label = "SSE")
		plotline, caplines, barlinecols = ax.errorbar(range(1, max_iter), model_mean, yerr = model_epsilon_std, fmt = 'ko', capsize = 5.0, label = r"$\mu + \varepsilon \sigma$", lolims = True)
		caplines[0].set_marker('_')
		ax.set(xlabel = r"Iteration $\tau$", ylabel = "Error", yscale = "log",
			# title = self.optimizer_method + " - Experiment " + str(self.experiment_id),
			xticks = range(max_iter)
		)
		ax.legend()
		
		# label_id = 0
		# for rect in rects[:-1]:
		# 	label_id += 1
		# 	ax.annotate(int(n_terms[label_id-1]),
		# 		xy = (rect.get_x() + rect.get_width() / 2, rect.get_height()),
		# 		xytext = (0, 3),
		# 		textcoords = "offset points",
		# 		ha = 'center', va = 'bottom')

		plt.savefig(os.path.join("output", "SSExIt_experiment" + str(self.experiment_id) + ".png"), bbox_inches = 'tight')
		plt.close()

	def plot_sse_correlation(self, width = 0.25):
		model_sse_deriv = self.model.get_SSE()
		model_sse_data = self.model.get_SSE_data()
		max_iter = len(model_sse_deriv)
		x = np.arange(max_iter)

		fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
		rects1 = ax.bar(x - width/2, model_sse_deriv, width = width, color = "b", alpha = 0.5, label = "Derivative")
		rects2 = ax.bar(x + width/2, model_sse_data, width = width, color = "g", alpha = 0.5, label = "Data")
		ax.set(xlabel = r"Iteration $\tau$", ylabel = "Error", yscale = "log",
			xticks = range(max_iter)
		)
		ax.legend()

		# ax.plot(model_sse_deriv, model_sse_data, "ko", alpha = 0.5, markersize = 3)
		# ax.set(xlabel = "Derivative", ylabel = "Data", xscale = "log", yscale = "log")

		plt.savefig(os.path.join("output", "CorrxIt_experiment" + str(self.experiment_id) + ".png"), bbox_inches = 'tight')
		plt.close()

	def plot_relative_error(self, width = 0.25):
		model_relative_error_deriv = self.model.get_relative_error()
		max_iter = len(model_relative_error_deriv)
		x = np.arange(1, max_iter+1)

		fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
		rects1 = ax.bar(x, model_relative_error_deriv, width = width, color = "b", alpha = 0.5, label = "Derivative")
		ax.set(xlabel = r"Iteration $\tau$", ylabel = "Relative Error", yscale = "log",
			# title = self.optimizer_method + " - Experiment " + str(self.experiment_id),
			xticks = range(1, max_iter+1)
		)
		ax.legend()

		plt.savefig(os.path.join("output", "RelErrorxIt_experiment" + str(self.experiment_id) + ".png"), bbox_inches = 'tight')
		plt.close()

	def plot_Ftest(self, width = 0.25):
		model_Ftest_deriv = self.model.get_Ftest()
		# model_Ftest_data = self.model.get_Ftest_data()
		max_iter = len(model_Ftest_deriv)
		x = np.arange(2, max_iter+2)

		fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
		rects1 = ax.bar(x, model_Ftest_deriv, width = width, color = "b", alpha = 0.5, label = "Derivative")
		# rects2 = ax.bar(x + width/2, model_Ftest_data, width = width, color = "g", alpha = 0.5, label = "Data")
		ax.set(xlabel = r"Iteration $\tau$", ylabel = "F-test", yscale = "log",
			# title = self.optimizer_method + " - Experiment " + str(self.experiment_id),
			xticks = range(2, max_iter+2)
		)
		ax.legend()

		# ax.plot(model_Ftest_deriv, model_Ftest_data, "ko", alpha = 0.5, markersize = 3)
		# ax.set(xlabel = "Derivative", ylabel = "Data", xscale = "log", yscale = "log")

		plt.savefig(os.path.join("output", "FtestxIt_experiment" + str(self.experiment_id) + ".png"), bbox_inches = 'tight')
		plt.close()

	def plot_qoi(self):
		model_num_eval = self.model.get_num_eval()
		max_iter = len(model_num_eval)
		model_history = self.model.get_history()

		n_terms = np.zeros(len(model_history)-1)
		for it in range(1, len(model_history)):
			n_terms[it-1] = np.count_nonzero(model_history[it] != 0.0)

		fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
		rects = ax.bar(range(max_iter), model_num_eval, width = 0.25, color = "b", alpha = 0.5)
		ax.set(xlabel = r"Iteration $\tau$", ylabel = "Number of QoI evaluations",
			title = # self.optimizer_method + " - Experiment " + str(self.experiment_id) + "\n"
			"Total number of QoI evaluations = " + str(sum(model_num_eval)),
			xticks = range(max_iter)
		)
		
		label_id = 0
		for rect in rects:
			label_id += 1
			ax.annotate(int(n_terms[label_id-1]),
				xy = (rect.get_x() + rect.get_width() / 2, rect.get_height()),
				xytext = (0, 3),
				textcoords = "offset points",
				ha = 'center', va = 'bottom')

		plt.savefig(os.path.join("output", "QoIxIt_experiment" + str(self.experiment_id) + ".png"), bbox_inches = 'tight')
		plt.close()

	def plot_musig(self):
		model_mu_star = self.model.get_mu_star()
		model_sigma = self.model.get_sigma()
		model_param_min = self.model.get_param_min()
		model_history = self.model.get_history()

		annotate_labels = np.empty((len(self.model.feature_names), len(self.model.get_feature_names())), dtype = object)
		for i, state_var in enumerate(self.model.feature_names):
			for j, term in enumerate(self.model.get_feature_names()):
				annotate_labels[i, j] = r"$(" + state_var + "," + term + ")$"

		for it in range(len(model_mu_star)):
			for sa_time in range(model_mu_star[it].shape[0]):
				fig, ax = plt.subplots(1, 1, figsize = (10, 10), dpi = 300)
				ax.plot(model_mu_star[it][sa_time], model_sigma[it][sa_time], "bo", alpha = 0.5, markersize = 5)
				if it < len(model_param_min):
					ax.plot(model_mu_star[it][sa_time][model_param_min[it]], model_sigma[it][sa_time][model_param_min[it]], "ro", alpha = 0.5, markersize = 5)
				ax.set(xlabel = r"$\mu_{i}^{*}$", ylabel = r"$\sigma_{i}$",
					# title = self.optimizer_method + " - Experiment " + str(self.experiment_id) + ", Iteration " + str(it) + ", Time " + str(sa_time)
				)

				coef = model_history[it+1]
				ind = coef != 0.0

				label_id = 0
				for i, j in zip(model_mu_star[it][sa_time], model_sigma[it][sa_time]):
					label_id += 1
					ax.annotate(annotate_labels[ind][label_id-1], xy = (i, j), xytext = (-12, 7), textcoords = 'offset points')

				plt.savefig(os.path.join("output", "musig_experiment" + str(self.experiment_id) + "_it" + str(it) + "_t" + str(sa_time) + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_mu_star(self):
		model_mu_star = self.model.get_mu_star()
		model_param_min = self.model.get_param_min()
		model_history = self.model.get_history()

		labels = np.empty(len(self.model.get_feature_names()), dtype = object)
		for i, term in enumerate(self.model.get_feature_names()):
			labels[i] = r"$" + term + "$"

		for it in range(len(model_mu_star)):
			coef = model_history[it+1]
			ind = coef != 0.0

			for sa_time in range(model_mu_star[it].shape[0]):
				mu_star = np.zeros(coef.shape)
				mu_star[ind] = model_mu_star[it][sa_time]

				if it < len(model_param_min):
					ind_param_min = np.full(coef.shape[0]*coef.shape[1], False)
					nonzero = -1
					for i in range(len(ind.flatten())):
						if ind.flatten()[i]:
							nonzero += 1
							if nonzero in model_param_min[it]:
								ind_param_min[i] = True
					ind_param_min = ind_param_min.reshape(coef.shape)

					mu_star_param_min = np.zeros(coef.shape)
					mu_star_param_min[ind_param_min] = model_mu_star[it][sa_time][model_param_min[it]]

				fig, axs = plt.subplots(1, len(self.model.feature_names), figsize = (15, 7.5/len(self.model.feature_names)), dpi = 300)
				for i, state_var in enumerate(self.model.feature_names):
					axs[i].bar(range(len(labels)), mu_star[i,:] - mu_star_param_min[i,:], width = 0.25, color = "b", alpha = 0.5)
					axs[i].bar(range(len(labels)), mu_star_param_min[i,:], width = 0.25, color = "r", alpha = 0.5)
					axs[i].set(xlabel = r"$" + state_var + "'(t)$", ylabel = r"$\mu_{i}^{*}$", yscale = "log",
						xticks = range(len(labels)),
						xticklabels = labels
					)

				fig.suptitle(r"$\tau = " + str(it) + "$", fontsize = 16, y = 0.95)
				plt.savefig(os.path.join("output", "mu_star_experiment" + str(self.experiment_id) + "_it" + str(it) + "_t" + str(sa_time) + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_sigma(self):
		model_sigma = self.model.get_sigma()
		model_param_min = self.model.get_param_min()
		model_history = self.model.get_history()

		labels = np.empty(len(self.model.get_feature_names()), dtype = object)
		for i, term in enumerate(self.model.get_feature_names()):
			labels[i] = r"$" + term + "$"

		for it in range(len(model_sigma)):
			coef = model_history[it+1]
			ind = coef != 0.0

			for sa_time in range(model_sigma[it].shape[0]):
				sigma = np.zeros(coef.shape)
				sigma[ind] = model_sigma[it][sa_time]

				if it < len(model_param_min):
					ind_param_min = np.full(coef.shape[0]*coef.shape[1], False)
					nonzero = -1
					for i in range(len(ind.flatten())):
						if ind.flatten()[i]:
							nonzero += 1
							if nonzero in model_param_min[it]:
								ind_param_min[i] = True
					ind_param_min = ind_param_min.reshape(coef.shape)

					sigma_param_min = np.zeros(coef.shape)
					sigma_param_min[ind_param_min] = model_sigma[it][sa_time][model_param_min[it]]

				fig, axs = plt.subplots(1, len(self.model.feature_names), figsize = (15, 7.5/len(self.model.feature_names)), dpi = 300)
				for i, state_var in enumerate(self.model.feature_names):
					axs[i].bar(range(len(labels)), sigma[i,:] - sigma_param_min[i,:], width = 0.25, color = "b", alpha = 0.5)
					axs[i].bar(range(len(labels)), sigma_param_min[i,:], width = 0.25, color = "r", alpha = 0.5)
					axs[i].set(xlabel = r"$" + state_var + "'(t)$", ylabel = r"$\sigma_{i}$", yscale = "log",
						xticks = range(len(labels)),
						xticklabels = labels
					)

				fig.suptitle(r"$\tau = " + str(it) + "$", fontsize = 16, y = 0.95)
				plt.savefig(os.path.join("output", "sigma_experiment" + str(self.experiment_id) + "_it" + str(it) + "_t" + str(sa_time) + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_ST(self):
		model_mu_star = self.model.get_mu_star()
		model_sigma = self.model.get_sigma()
		model_param_min = self.model.get_param_min()
		model_history = self.model.get_history()

		labels = np.empty(len(self.model.get_feature_names()), dtype = object)
		for i, term in enumerate(self.model.get_feature_names()):
			labels[i] = r"$" + term + "$"

		for it in range(len(model_mu_star)):
			coef = model_history[it+1]
			ind = coef != 0.0

			for sa_time in range(model_mu_star[it].shape[0]):
				ST = np.zeros(coef.shape)
				ST[ind] = np.sqrt(model_mu_star[it][sa_time]**2.0 + model_sigma[it][sa_time]**2.0)

				if it < len(model_param_min):
					ind_param_min = np.full(coef.shape[0]*coef.shape[1], False)
					nonzero = -1
					for i in range(len(ind.flatten())):
						if ind.flatten()[i]:
							nonzero += 1
							if nonzero in model_param_min[it]:
								ind_param_min[i] = True
					ind_param_min = ind_param_min.reshape(coef.shape)

					ST_param_min = np.zeros(coef.shape)
					ST_param_min[ind_param_min] = np.sqrt(model_mu_star[it][sa_time][model_param_min[it]]**2.0 + model_sigma[it][sa_time][model_param_min[it]]**2.0)

				fig, axs = plt.subplots(1, len(self.model.feature_names), figsize = (15, 7.5/len(self.model.feature_names)), dpi = 300)
				for i, state_var in enumerate(self.model.feature_names):
					axs[i].bar(range(len(labels)), ST[i,:] - ST_param_min[i,:], width = 0.25, color = "b", alpha = 0.5)
					axs[i].bar(range(len(labels)), ST_param_min[i,:], width = 0.25, color = "r", alpha = 0.5)
					axs[i].set(xlabel = r"$" + state_var + "'(t)$", ylabel = r"$\mathcal{S}_{i}$", yscale = "log",
						xticks = range(len(labels)),
						xticklabels = labels
					)

				fig.suptitle(r"$\tau = " + str(it) + "$", fontsize = 16, y = 0.95)
				plt.savefig(os.path.join("output", "ST_experiment" + str(self.experiment_id) + "_it" + str(it) + "_t" + str(sa_time) + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_simulation(self, X = None, t = None, X0 = None, precision = 3):
		coef = self.model.coefficients()
		model_history = self.model.get_history()
		markers = ["o", "^", "s", "p", "P", "*", "X", "d"]

		for it in range(1, len(model_history)):
			k = np.count_nonzero(model_history[it] != 0.0)
			self.model.coefficients(model_history[it])
			simulation = self.model.simulate(X0, t = t)

			equations = ""
			for i, feature_name in enumerate(self.model.feature_names):
				equations += feature_name + "' = " + self.model.equations(precision = precision)[i] + "\n"

			fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
			for i, feature_name in enumerate(self.model.feature_names):
				ax.plot(t, X[:,i], "k" + markers[i], label = r"Data $" + self.model.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
			for i, feature_name in enumerate(self.model.feature_names):
				ax.plot(t, simulation[:,i], label = r"Model $" + self.model.feature_names[i] + "(t)$", alpha = 1.0, linewidth = 1)
			ax.set(xlabel = r"Time $t$", ylabel = r"$X(t)$",
				# title = equations
				# + "Initial condition = " + str(X0)
			)
			ax.legend()
			# fig.suptitle(self.optimizer_method + " - Experiment " + str(self.experiment_id) + ", Iteration " + str(it-1), fontsize = 16, y = 0.99)
			# fig.show()
			plt.savefig(os.path.join("output", "sim_experiment" + str(self.experiment_id) + "_it" + str(it-1) + ".png"), bbox_inches = 'tight')
			plt.close()

		self.model.coefficients(coef)

	def plot_derivative(self, X = None, t = None):
		X_dot_data = self.model.differentiate(X, t)
		X_dot_model = self.model.get_X_dot_model()
		markers = ["o", "^", "s", "p", "P", "*", "X", "d"]

		for it in range(len(X_dot_model)):
			fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
			for i, feature_name in enumerate(self.model.feature_names):
				ax.plot(t, X_dot_data[:,i], "k" + markers[i], label = r"Data $" + self.model.feature_names[i] + "'(t)$", alpha = 0.5, markersize = 3)
			for i, feature_name in enumerate(self.model.feature_names):
				ax.plot(t, X_dot_model[it][:,i], label = r"Model $" + self.model.feature_names[i] + "'(t)$", alpha = 1.0, linewidth = 1)
			ax.set(xlabel = r"Time $t$", ylabel = r"$X'(t)$")
			ax.legend()
			plt.savefig(os.path.join("output", "deriv_experiment" + str(self.experiment_id) + "_it" + str(it) + ".png"), bbox_inches = 'tight')
			plt.close()