TEST_CONFIG=./configs/testConfigs.json
SL_CONFIG=./configs/sLConfigs.json
ML_CONFIG=./configs/mLConfigs.json
# CONFIG=($SL_CONFIG $ML_CONFIG)
CONFIG=($SL_CONFIG)
# P=v100
# P=elephant
# CONFIG=($ML_CONFIG)
# CONFIG=($TEST_CONFIG)

FTR=(rd)
NORM=(L T)

for norm in "${NORM[@]}"; do
    if [ $norm == L ]
    then
        NORMDIM=(2)
    elif [ $norm == T ]
    then
        NORMDIM=(8 16)
    else
        raise error "norm funtion not recognized"
    fi
    for config in "${CONFIG[@]}"; do
        for ftr in "${FTR[@]}"; do
            for normDim in "${NORMDIM[@]}"; do
                ./scripts/main.sh \
                    srun -p elephant\
                    --config $config \
                    --ftr $ftr \
                    --norm $norm \
                    --normDim $normDim
            done
        done
    done
done