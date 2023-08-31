# Create icub_config.yaml config files for each CUB run that has a pred_config.yaml
shopt -s nullglob

RUNS=(logs/cub/run-*)

for run in ${RUNS[@]}
do
    if [ -f "$run/pred_config.yaml" ] && [ ! -f "$run/icub_config.yaml" ]; then
        echo "Creating $run/icub_config.yaml"
        # copy pred_config.yaml and change path and dataset from CUB to iCub
        cp "$run/pred_config.yaml" "$run/icub_config.yaml"
        sed -i 's/CUB_200_2011/inat-cub/' "$run/icub_config.yaml"
        sed -i 's/data\.CUB/data.InatCUB/' "$run/icub_config.yaml"
    fi
done