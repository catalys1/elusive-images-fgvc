# Run prediction extraction for iCub
shopt -s nullglob

DRY=${1:-"0"}

RUNS=(logs/cub/run-*)

for run in ${RUNS[@]}
do
    if [ -f "$run/icub_config.yaml" ] && [ ! -f "$run/icub_preds.pth" ]; then
        if [ $DRY == "1" ]; then
            echo $run
        else
            echo "Get preds for $run"
            python run.py predict -c "$run/icub_config.yaml"
        fi
    fi
done
