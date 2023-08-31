# Run prediction extraction.
# Runs the "predict" routine on every run that has a pred_config.yaml but no
# preds.pth in the run directory.
# Produces preds.pth in the run directory.
#
# If run as
# $ bash get_predictions 1
# then does a dry run, only printing the runs that would be activated.
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
