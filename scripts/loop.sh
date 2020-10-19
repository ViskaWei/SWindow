TEST_CONFIG=./configs/testConfigs.json
CSL_CONFIG=./configs/csLConfigs.json
ML_CONFIG=./configs/mLConfigs.json
# CONFIG=($CSL_CONFIG $ML_CONFIG)
CONFIG=($TEST_CONFIG)


for config in "${CONFIG[@]}"; do
    ./scripts/main.sh \
        --config $config
done