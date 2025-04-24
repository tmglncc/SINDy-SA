# SINDy-SA Framework

Machine learning methods have revolutionized studies in several areas of knowledge, helping to understand and extract information from experimental data. Recently, these data-driven methods have also been used to discover structures of mathematical models. The sparse identification of nonlinear dynamics (SINDy) method has been proposed with the aim of identifying nonlinear dynamical systems, assuming that the equations have only a few important terms that govern the dynamics. By defining a library of possible terms, the SINDy approach solves a sparse regression problem by eliminating terms whose coefficients are smaller than a threshold. However, the choice of this threshold is decisive for the correct identification of the model structure. In this work, we build on the SINDy method by integrating it with a global sensitivity analysis (SA) technique that allows to hierarchize terms according to their importance in relation to the desired quantity of interest, thus circumventing the need to define the SINDy threshold. The proposed SINDy-SA framework also includes the formulation of different experimental settings, recalibration of each identified model, and the use of model selection techniques to select the best and most parsimonious model. We investigate the use of the proposed SINDy-SA framework in a variety of applications. We also compare the results against the original SINDy method. The results demonstrate that the SINDy-SA framework is a promising methodology to accurately identify interpretable data-driven models. 

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

Naozuka, G.T., Rocha, H.L., Silva, R.S. _et al._ SINDy-SA framework: enhancing nonlinear system identification with sensitivity analysis. _Nonlinear Dyn_ **110**, 2589–2609 (2022). [https://doi.org/10.1007/s11071-022-07755-2](https://doi.org/10.1007/s11071-022-07755-2)

Naozuka, G.T.; Rocha, H.L.; Silva, R.S.; Almeida, R.C. SINDy-SA, 2022. Version 1.0. Available online: [https://github.com/tmglncc/SINDy-SA](https://github.com/tmglncc/SINDy-SA) (accessed on 24 January 2022), doi: [10.5281/zenodo.5931191](https://doi.org/10.5281/zenodo.5931191).
