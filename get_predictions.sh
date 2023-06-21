shopt -s nullglob

DRY=${1:-"0"}

DATASETS=( aircraft cars cub fungi nabirds )

for dset in ${DATASETS[@]}
do
    RUNS=(logs/$dset/run-*)

    for run in ${RUNS[@]}
    do
        if [ -f "$run/pred_config.yaml" ] && [ ! -f "$run/preds.pth" ]; then
            if [ $DRY == "1" ]; then
                echo $run
            else
                echo "Get preds for $run"
                python run.py predict -c $run/pred_config.yaml
            fi
        fi
    done
done