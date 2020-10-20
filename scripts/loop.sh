TEST_CONFIG=./configs/testConfigs.json
CSL_CONFIG=./configs/csLConfigs.json
ML_CONFIG=./configs/mLConfigs.json
CONFIG=($CSL_CONFIG $ML_CONFIG)
FTR=(rd src)
NORM=(L T)
# CONFIG=($TEST_CONFIG)

for norm in "${NORM[@]}"; do
    if [ $norm == L ]
    then
        NORMDIM=(1 2)
        p=elephant
    elif [ $norm == T ]
    then
        NORMDIM=(8 16)
        p=v100
    else
        raise error "norm funtion not recognized"
    fi
    for config in "${CONFIG[@]}"; do
        for ftr in "${FTR[@]}"; do
            for normDim in "${NORMDIM[@]}"; do
                ./scripts/main.sh \
                    sbatch -p $p\
                    --config $config \
                    --ftr $ftr \
                    --norm $norm \
                    --normDim $normDim
            done
        done
    done
done