# SymNormSlidingWindows
Vladimir Braverman, Samson Zhou
## Run from terminal
    $ ./main.sh 

# Use with SLURM

## Run job interactively

	$ srun -p v100 --gpus=1 --cpu-per-task=12 ./scripts/train.sh ...
	
* -p: name of the partition of servers with GPUs
* --gpus: number of GPUs to allocate to learning process
* --cpu-per-task: number of CPU cores to allocate to learning process

## Submit to the queue


