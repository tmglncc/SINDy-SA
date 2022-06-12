import numpy as np
import os

class ModelSelection:
	def __init__(self, model_set = None, n = None):
		self.model_set = model_set
		self.n = n

		if model_set is not None:
			self.num_models = len(self.model_set)

			self.k = np.zeros(self.num_models)
			self.SSE = np.zeros(self.num_models)

			self.AIC = np.zeros(self.num_models)
			self.Delta_AIC = np.zeros(self.num_models)
			self.like = np.zeros(self.num_models)
			self.AIC_weights = np.zeros(self.num_models)
			self.AIC_evid_ratio = np.zeros(self.num_models)

			self.AICc = np.zeros(self.num_models)
			self.Delta_AICc = np.zeros(self.num_models)
			self.likec = np.zeros(self.num_models)
			self.AICc_weights = np.zeros(self.num_models)
			self.AICc_evid_ratio = np.zeros(self.num_models)

			self.BIC = np.zeros(self.num_models)
			self.Delta_BIC = np.zeros(self.num_models)
			self.BIC_prob = np.zeros(self.num_models)

	def compute_k(self):
		for model_id, model in enumerate(self.model_set):
			# self.k[model_id] = np.count_nonzero(np.absolute(model.coefficients()) >= 5.0e-4)
			self.k[model_id] = np.count_nonzero(model.coefficients() != 0.0)

	def compute_SSE(self, target, predicted):
		squared_errors = (target - predicted)**2.0
		return np.sum(squared_errors)

	def set_model_SSE(self, model_id, SSE):
		self.SSE[model_id] = SSE

	def compute_AIC(self):
		self.AIC = self.n*np.log(self.SSE/self.n) + 2.0*self.k
		AICmin = np.amin(self.AIC)
		self.Delta_AIC = self.AIC - AICmin
		self.like = np.exp(-0.5*self.Delta_AIC)
		likesum = np.sum(self.like)
		self.AIC_weights = self.like/likesum
		self.best_AIC_model = np.argmax(self.AIC_weights)
		self.AIC_evid_ratio = self.AIC_weights[self.best_AIC_model]/self.AIC_weights
		return self.best_AIC_model

	def compute_AICc(self):
		self.AICc = self.n*np.log(self.SSE/self.n) + 2.0*self.k  + (2.0*self.k*(self.k + 1.0))/(self.n - self.k - 1.0)
		AICcmin = np.amin(self.AICc)
		self.Delta_AICc = self.AICc - AICcmin
		self.likec = np.exp(-0.5*self.Delta_AICc)
		likecsum = np.sum(self.likec)
		self.AICc_weights = self.likec/likecsum
		self.best_AICc_model = np.argmax(self.AICc_weights)
		self.AICc_evid_ratio = self.AICc_weights[self.best_AICc_model]/self.AICc_weights
		return self.best_AICc_model

	def compute_BIC(self):
		self.BIC = self.n*np.log(self.SSE/self.n) + self.k*np.log(self.n)
		BICmin = np.amin(self.BIC)
		self.Delta_BIC = self.BIC - BICmin
		BICsum = np.sum(np.exp(-0.5*self.Delta_BIC))
		self.BIC_prob = np.exp(-0.5*self.Delta_BIC)/BICsum
		self.best_BIC_model = np.argmax(self.BIC_prob)
		return self.best_BIC_model

	def write_output(self, filename = os.path.join("output", "output.dat")):
		file = open(filename, "w")

		file.write("Modelo \t SSE \t AIC \t AICc \t Delta_AIC \t Delta_AICc \t Likelihood \t Likelihood_c \t AIC_weights \t AICc_weights \t AIC_evid_ratio \t AICc_evid_ratio \t BIC \t Delta_BIC \t BIC_prob\n")

		for model_id, model in enumerate(self.model_set):
			file.write(str(model_id+1) + "\t"
				+ str(self.SSE[model_id]) + "\t"
				+ str(self.AIC[model_id]) + "\t"
				+ str(self.AICc[model_id]) + "\t"
				+ str(self.Delta_AIC[model_id]) + "\t"
				+ str(self.Delta_AICc[model_id]) + "\t"
				+ str(self.like[model_id]) + "\t"
				+ str(self.likec[model_id]) + "\t"
				+ str(self.AIC_weights[model_id]) + "\t"
				+ str(self.AICc_weights[model_id]) + "\t"
				+ str(self.AIC_evid_ratio[model_id]) + "\t"
				+ str(self.AICc_evid_ratio[model_id]) + "\t"
				+ str(self.BIC[model_id]) + "\t"
				+ str(self.Delta_BIC[model_id]) + "\t"
				+ str(self.BIC_prob[model_id]) + "\n"
			)

		file.close()

	def write_AICc_weights(self, filename = os.path.join("output", "weights.dat")):
		file = open(filename, "w")

		for model_id, model in enumerate(self.model_set):
			file.write(str(model_id+1) + " "
				+ str(self.SSE[model_id]) + " "
				+ str(self.AICc_weights[model_id]) + "\n"
			)

		file.close()

	def write_pareto_curve(self, optimizer_method, filename = os.path.join("output", "pareto.dat")):
		file = open(filename, "w")

		file.write(optimizer_method + " "
			+ str(self.best_AICc_model+1) + " "
			+ str(self.num_models) + "\n"
		)
		for model_id, model in enumerate(self.model_set):
			file.write(str(model_id+1) + " "
				+ str(self.k[model_id]) + " "
				+ str(self.SSE[model_id]) + "\n"
			)

		file.close()

	def read_pareto_curve(self, filename = os.path.join("output", "pareto.dat")):
		with open(filename, "r") as reader:
		    for i, line in enumerate(reader):
		        split_line = line.split(" ")
		        if i == 0:
		        	optimizer_method = split_line[0]
		        	self.best_AICc_model = int(split_line[1])-1
		        	self.num_models = int(split_line[2])

		        	self.k = np.zeros(self.num_models)
		        	self.SSE = np.zeros(self.num_models)
		        else:
		        	model_id = int(split_line[0])-1
		        	self.k[model_id] = float(split_line[1])
		        	self.SSE[model_id] = float(split_line[2])

		return optimizer_method