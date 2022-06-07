import numpy as np
import pysindy_local2 as ps
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from ModelPlots import ModelPlots
from ModelCalibration import ModelCalibration
from ModelSelection import ModelSelection
from DataDenoising import DataDenoising

def logistic_model(X, t, a, b):
	dXdt = a*X*(1.0 - X/b)
	return dXdt

def is_new_model(model_set, model, n_vars, precision):
	for model_element in model_set:
		flag = True
		for i in range(n_vars):
			if model_element.equations(precision = precision)[i] != model.equations(precision = precision)[i]:
				flag = False
				break

		if flag:
			return False

	return True

experiment_id = 0

# Train data parameters
a = 0.028
b = 8472.914

X0 = 245.185
X0_tests = [245.185]

t0 = 0.0
tf = 300.0
t_steps = 300

# Method parameters
fd_order = 2
poly_degrees = range(2, 6)
fourier_nfreqs = range(1, 2)
optimizer_method = "STLSQ+SA"
precision = 9

plot_sse = False
plot_qoi = False
plot_ST = False
plot_musig = False
plot_simulation = False
calibration_mode = "LM"

stlsq_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
stlsq_thresholds = [1.0e-7, 1.0e-6, 1.0e-5]

# Generate train data
t = np.linspace(t0, tf, t_steps)
X = odeint(logistic_model, X0, t, args=(a, b))

dd = DataDenoising(X, t, ["N"])
dd.plot_sma([5, 10, 20])
dd.plot_ema([0.1, 0.2, 0.3], [False])
dd.plot_l2r([10.0, 100.0, 1000.0])
dd.plot_tvr([0.001], [0.25, 0.5, 1.0])

model_set = []
for poly_degree in poly_degrees:
	for fourier_nfreq in fourier_nfreqs:
		for stlsq_alpha in stlsq_alphas:
			for stlsq_threshold in stlsq_thresholds:
				experiment_id += 1
				print("Experimento " + str(experiment_id) 
					+ ": Grau = " + str(poly_degree) 
					+ ", Frequência = " + str(fourier_nfreq) 
					+ ", alpha = " + str(stlsq_alpha)
					+ ", threshold = " + str(stlsq_threshold) + "\n"
				)

				# Define method properties
				differentiation_method = ps.FiniteDifference(order = fd_order)
				# differentiation_method = ps.SmoothedFiniteDifference()
				feature_library = ps.PolynomialLibrary(degree = poly_degree) # + ps.FourierLibrary(n_frequencies = fourier_nfreq)
				optimizer = ps.STLSQ(
					alpha = stlsq_alpha,
					threshold = stlsq_threshold,
					fit_intercept = False
				)

				# Compute sparse regression
				model = ps.SINDy(
					differentiation_method = differentiation_method,
					feature_library = feature_library,
					optimizer = optimizer,
					feature_names = ["N"]
				)
				model.fit(X, t = t)
				model.print(precision = precision)
				print("\n")

				# Generate model plots
				mp = ModelPlots(model, optimizer_method, experiment_id)
				if plot_sse:
					mp.plot_sse()
				if plot_qoi:
					mp.plot_qoi()
				if plot_ST:
					mp.plot_mu_star()
					mp.plot_sigma()
					mp.plot_ST()
				if plot_musig:
					mp.plot_musig()
				if plot_simulation:
					mp.plot_simulation(X, t, X0, precision = precision)

				# Add model to the set of models
				if not model_set or is_new_model(model_set, model, len(model.feature_names), precision):
					model_set.append(model)

# Compute number of terms
ms = ModelSelection(model_set, t_steps)
ms.compute_k()

for model_id, model in enumerate(model_set):
	print("Modelo " + str(model_id+1) + "\n")
	model.print(precision = precision)
	print("\n")

	sse_sum = 0.0
	for init_cond_id, X0_test in enumerate(X0_tests):
		# Generate test data
		X_test = odeint(logistic_model, X0_test, t, args=(a, b))

		dd = DataDenoising(X_test, t, model.feature_names)

		# Compute derivative
		X_dot_test = model.differentiate(X_test, t)
		dd.plot_derivative(X_dot_test, t, init_cond_id, X0_test)

		# Simulate with another initial condition
		if calibration_mode is None:
			simulation = model.simulate([X0_test], t = t)
		elif calibration_mode == "LM":
			mc = ModelCalibration(model, model_id, X_test, t, X0_test, init_cond_id)
			mc.levenberg_marquardt()
			model.print(precision = precision)
			print("\n")

			simulation = model.simulate([X0_test], t = t)
		elif calibration_mode == "Bayes":
			mc = ModelCalibration(model, model_id, X_test, t, X0_test, init_cond_id)
			mc.bayesian_calibration()
			mc.traceplot()
			mc.plot_posterior()
			mc.plot_pair()
			X0_test = mc.summary()
			print("\n")
			model.print(precision = precision)
			print("\n")

			simulation, simulation_min, simulation_max = mc.get_simulation()

		# Generate figures
		plt.rcParams.update({'font.size': 20})
		fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
		ax.plot(t, X_test[:,0], "ko", label = r"Data $x(t)$", alpha = 0.3, markersize = 5)
		ax.plot(t, simulation[:,0], "k-", label = r"Model $x(t)$", alpha = 1.0, linewidth = 1)
		if calibration_mode == "Bayes":
			ax.fill_between(t, simulation_min[:,0], simulation_max[:,0], color = "k", alpha = 0.4)
		ax.set(xlabel = r"Time $t$", ylabel = r"$X(t)$",
			# title = "N' = " + model.equations(precision = precision)[0] + "\n"
			# + "Initial condition = " + str(X0_test)
		)
		ax.legend()
		# fig.suptitle(optimizer_method + " - Model " + str(model_id+1) + ", Initial condition " + str(init_cond_id), fontsize = 16, y = 0.99)
		# fig.show()
		plt.savefig(os.path.join("output", "model" + str(model_id+1) + "_ic" + str(init_cond_id) + ".png"), bbox_inches = 'tight')
		plt.close()

		# Compute SSE
		sse_sum += ms.compute_SSE(X_test.reshape(simulation.shape), simulation)

	# Set mean SSE to the model
	ms.set_model_SSE(model_id, sse_sum/len(X0_tests))

# Compute AIC and AICc
best_AIC_model = ms.compute_AIC()
best_AICc_model = ms.compute_AICc()
best_BIC_model = ms.compute_BIC()

# Get best model
print("Melhor modelo AIC = " + str(best_AIC_model+1) + "\n")
print("Melhor modelo AICc = " + str(best_AICc_model+1) + "\n")
print("Melhor modelo BIC = " + str(best_BIC_model+1) + "\n")

# Write results
ms.write_output()
ms.write_AICc_weights()
ms.write_pareto_curve(optimizer_method)
