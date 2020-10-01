#!/bin/bash
START_EPOCH=0
NUM_EPOCHS=150
LR=0.001
PATIENCE=2
BATCH_SIZE=15
NUM_WORKERS=25
NUM_GPUS=1
GPUS=0

SSD_LOCATION="/media/SSD3/carpos"
DATASET="RHPE"
EXPERIMENT_NAME="/best_exp/"$DATASET


DATA_TRAIN=$SSD_LOCATION"/data/"$DATASET"/train/images/"
ANN_PATH_TRAIN=$SSD_LOCATION"/data/"$DATASET"/new_Chronological_train.csv"
# ANN_PATH_TRAIN=$SSD_LOCATION"/data/"$DATASET"/train/annotations/boneage.csv"
HEATMAPS_TRAIN=$SSD_LOCATION"/data/"$DATASET"/train/heatmaps/"
ROIS_PATH_TRAIN=$SSD_LOCATION"/data/"$DATASET"/train/annotations/anatomical_ROIs.json"

mkdir -p $HEATMAPS_TRAIN

DATA_VAL=$SSD_LOCATION"/data/"$DATASET"/val/images/"
HEATMAPS_VAL=$SSD_LOCATION"/data/"$DATASET"/val/heatmaps/"
ANN_PATH_VAL=$SSD_LOCATION"/data/"$DATASET"/new_Chronological_val.csv"
# ANN_PATH_VAL=$SSD_LOCATION"/data/"$DATASET"/val/annotations/boneage.csv"
ROIS_PATH_VAL=$SSD_LOCATION"/data/"$DATASET"/val/annotations/anatomical_ROIs.json"
mkdir -p $HEATMAPS_VAL

SAVE_FOLDER=$SSD_LOCATION"/experiments"$EXPERIMENT_NAME

mkdir -p $SAVE_FOLDER

#SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_snapshot.pth"
SNAPSHOT=$SSD_LOCATION"/experiments/MICCAI2020/SIMBA/chronological-multiplier-relative/"$DATASET"/boneage_bonet_weights.pth"
OPTIM_SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_optim.pth"


CUDA_VISIBLE_DEVICES=$GPUS mpirun -np $NUM_GPUS -H localhost:$NUM_GPUS -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_CUDA_HOME=/usr/local/cuda-10.0 -mca pml ob1 -mca btl ^openib python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --cropped --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier #>> $SAVE_FOLDER"/log.txt"

#--eval-first