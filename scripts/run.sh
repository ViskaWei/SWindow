#!/bin/bash

# Work around issues with saving weights when running on multiple threads
# export HDF5_USE_FILE_LOCKING=FALSE
source /datascope/slurm/miniconda3/bin/activate ptorch

PARAMS=""
COMMAND="$1"
shift

# Get runmode

if [[ $1 == "sbatch" ]] || [[ $1 == "srun" ]]; then
    RUNMODE=$1
    shift
elif [[ $1 == "run" ]]; then
    RUNMODE=$1
    shift
else
    RUNMODE="run"
fi

# Parse slurm and other parameters

PYTHON_DEBUG=0
SBATCH_PARTITION='elephant'
SBATCH_MEM=32G
SBATCH_TIME="24:00:00"
SBATCH_GPUS=0
SBATCH_CPUS_PER_TASK=32

while (( "$#" )); do
    case "$1" in
        --debug)
            PYTHON_DEBUG=1
            shift
            ;;
        -p|--partition)
            SBATCH_PARTITION=$2
            shift 2
            ;;
        --mem)
            SBATCH_MEM=$2
            shift 2
            ;;
        -t|--time)
            SBATCH_TIME=$2
            shift 2
            ;;
        -G|--gpus)
            if [[ $RUNMODE != "run" ]]; then
                SBATCH_GPUS=$2
            else
                PARAMS="$PARAMS $1 $2"
            fi
            shift 2
            ;;
        -c|--cpus-per-task|--cpus)
            SBATCH_CPUS_PER_TASK=$2
            shift 2
            ;;
        --) # end argument parsing
            shift
            break
            ;;
        #-*|--*=) # unsupported flags
            #  echo "Error: Unsupported flag $1" >&2
            #  exit 1
            #  ;;
        *) # preserve all other arguments
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

if [[ $RUNMODE == "run" ]]; then
    if [[ $PYTHON_DEBUG == "1" ]]; then
        exec python -m ptvsd --host localhost --port 5678 --wait $COMMAND $PARAMS --debug
    else
        exec python $COMMAND $PARAMS
    fi
elif [[ $RUNMODE == "srun" ]]; then
    exec srun --partition $SBATCH_PARTITION \
              --gres gpu:$SBATCH_GPUS \
              --cpus-per-task $SBATCH_CPUS_PER_TASK \
              --mem $SBATCH_MEM \
              --time $SBATCH_TIME \
              python $COMMAND $PARAMS
elif [[ $RUNMODE == "sbatch" ]]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition $SBATCH_PARTITION
#SBATCH --gres gpu:$SBATCH_GPUS
#SBATCH --cpus-per-task $SBATCH_CPUS_PER_TASK
#SBATCH --mem $SBATCH_MEM
#SBATCH --time $SBATCH_TIME

set -e

out=slurm-\$SLURM_JOB_ID.out
srun python $COMMAND $PARAMS
outdir=\$(cat \$out | grep -Po 'Output directory is (\K.+)')
mv \$out \$outdir/slurm.out
EOF
else
    echo "Invalid RUNMODE: $RUNMODE"
    exit -1
fi
