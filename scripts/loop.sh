TEST_CONFIG=./configs/testConfigs.json
CSL_CONFIG=./configs/csLConfigs.json
ML_CONFIG=./configs/mLConfigs.json
CONFIG=($CSL_CONFIG $ML_CONFIG)
FTR=(rd src)
NORM=(L T)
# CONFIG=($TEST_CONFIG)


for config in "${CONFIG[@]}"; do
    for ftr in "${FTR[@]}"; do
        for norm in "${NORM[@]}"; do
            ./scripts/main.sh \
                --config $config \
                --ftr $ftr \
                --norm $norm
        done
    done
done