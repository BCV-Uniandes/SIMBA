#!/bin/bash
BATCH_SIZE=1
NUM_WORKERS=10
NUM_GPUS=1
GPUS=2

SSD_LOCATION="/media/SSD3/carpos"

DATASET="RHPE"

EXPERIMENT_NAME="/journal/bonet/"$DATASET

DATA_TEST=$SSD_LOCATION"/data/"$DATASET"/val/images/"
HEATMAPS_TEST=$SSD_LOCATION"/data/"$DATASET"/val/heatmaps/"
ANN_PATH_TEST=$SSD_LOCATION"/data/"$DATASET"/val/annotations/boneage_chronological.csv"
ROIS_PATH_TEST=$SSD_LOCATION"/data/"$DATASET"/val/annotations/anatomical_ROIs.json"
  
mkdir -p $HEATMAPS_TEST
  
SAVE_FOLDER=$SSD_LOCATION"/experiments"$EXPERIMENT_NAME
 
mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_weights.pth"
SAVE_FILE="validation_bestmodel.csv"

CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS --cropped #> $SAVE_FOLDER/"validation_bestmodel_log.txt"
