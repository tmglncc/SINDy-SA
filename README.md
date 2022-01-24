# SINDy-SA

In this work, we build on the sparse identification of nonlinear dynamics (SINDy) method by integrating it with a global sensitivity analysis (SA) technique to identify nonlinear dynamical systems. The SA technique allows us to classify terms according to their importance in relation to the desired quantity of interest and eliminates the need to define the SINDy threshold. We compare our proposed SINDy-SA approach with the original SINDy method employing them in a variety of applications whose simulated data have different behaviors, namely a prey-predator model, a logistic model calibrated using tumor growth data, a pendulum motion model, and a SIR compartmental model. For each application, we formulate different experimental settings and select the best model for both methods using model selection techniques based on information criteria. The proposed framework for solving the problem of identifying nonlinear dynamical systems is represented schematically in the following figure.

![Schematic representation of the process for solving the problem of identifying nonlinear dynamical systems, using the proposed SINDy-SA approach](https://drive.google.com/uc?export=view&id=15i_lycwttlDDCPhD_RduIcA-SjXViXz6)

## Requirements

Our experiments have been performed using **Python 3.8.2** and **Gnuplot 5.2**. The following modules are required to run both SINDy-SA and SINDy methods:
- NumPy 1.21.3 (https://numpy.org/)
- PySINDy 1.6.1 (https://pysindy.readthedocs.io/en/latest/)
- Matplotlib 3.4.3 (https://matplotlib.org/)
- SciPy 1.7.1 (https://scipy.org/)
- pandas 1.3.4 (https://pandas.pydata.org/)
- PyMC3 3.11.2 (https://docs.pymc.io/en/v3/)
- ArviZ 0.11.2 (https://arviz-devs.github.io/arviz/)
- tqdm 4.60.0 (https://tqdm.github.io/)
- Theano 1.0.5 (https://pypi.org/project/Theano/)
- PyLops 1.15.0 (https://pylops.readthedocs.io/en/latest/)
- plottools 0.2.0 (https://pypi.org/project/plottools/)

## Running SINDy-SA and SINDy applications

 1. Clone this repository directly from terminal:
	 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ git clone https://github.com/tmglncc/SINDy-SA.git`
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**OR**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Download the _.zip_ file and decompress it.

 2. Enter the directory of the application you would like to run and choose between SINDy-SA and SINDy methods.

 3. Clean the project's _output_ folder:
	 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ make clean`

 4. Run the project by writing the output logs to the terminal:
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ make`
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**OR**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run the project by writing the output logs to the _models.dat_ file:
	 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ make run_output`

 5. Enter the project's _output_ folder and check out the results.

## Cite as

Naozuka, G.T.; Rocha, H.L.; Silva, R.S.; Almeida, R.C. SINDy-SA, 2022. Version 1.0. Available online: [https://github.com/tmglncc/SINDy-SA](https://github.com/tmglncc/SINDy-SA) (accessed on 24 January 2022).
