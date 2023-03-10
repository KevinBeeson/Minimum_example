# Minimum_example
This is an example of my program's slowing down

All the dependencies are in the environment.yml file. Just create a new enviroment with the file using conda env create -f environment.yml the name is of the enviroment is payne_emcee

You can run the code by doing bash bash_looper_analysis.sh the out_put will be in run_out_puts

I've put 5 different examples 

170830002301099

170506006401016

170506006401012

170830002301102

170506006401388

You can change how many things you run by inputing these files into minimal_run.txt.

For this minimum example if I only run 1 file I get around 3-4 seconds per itartion but if I run all 5 at once I get 7-8 seconds per iteration so a doubling....

If you would like to look at which function takes the most time I would look at log_posterior in the Payne_minimal_analysis.py. You can also uncomment the lower part of the code if you would like to.

signal.fftconvolve and np.einsum takes doubles in time but I dont know why....
