#!/bin/bash
BATCH_SIZE=1
NUM_WORKERS=10
NUM_GPUS=1
GPUS=2

SSD_LOCATION="/media/SSD3/carpos"
DATASET="RHPE"
EXPERIMENT_NAME="MICCAI2020/SIMBA/chronological-multiplier-relative/"$DATASET

DATA_TEST=$SSD_LOCATION"/data/"$DATASET"/test/images/"
ANN_PATH_TEST=$SSD_LOCATION"/data/"$DATASET"/new_Chronological_test.csv"
HEATMAPS_TEST=$SSD_LOCATION"/data/"$DATASET"/test/heatmaps/"
ROIS_PATH_TEST=$SSD_LOCATION"/data/"$DATASET"/test/annotations/anatomical_ROIs.json"

#DATA_TEST= #Path to test images
#HEATMAPS_TEST= #Path to test heatmaps (Will be created automatically)
#ANN_PATH_TEST= #Path to csv annotations
#ROIS_PATH_TEST= #Path to json annotations of ROIs
  
mkdir -p $HEATMAPS_TEST
  
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME
 
mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_weights.pth"
SAVE_FILE="validation_bestmodel_2.csv"

CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS --cropped --chronological-age --relative-age --gender-multiplier --inference-only #> $SAVE_FOLDER/"validation_bestmodel_log.txt"
