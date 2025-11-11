#!/usr/bin/env bash

DATA_PATH=/p/realai/BioXplorer/CorNet_Model/data

# DATASET=EUR-Lex
# DATASET=AmazonCat-13K
# DATASET=Mesh-2022
# DATASET=Mesh-2022-100K
DATASET=Mesh-2022-100K-12Terms-1
#DATASET=Wiki-500K

#MODEL=XMLCNN
#MODEL=CorNetXMLCNN
#MODEL=BertXML
#MODEL=CorNetBertXML
#MODEL=MeSHProbeNet
MODEL=CorNetMeSHProbeNet
#MODEL=AttentionXML
#MODEL=CorNetAttentionXML

python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml

python evaluation.py \
--results $DATA_PATH/$DATASET/results/$MODEL-$DATASET-labels.npy \
--targets $DATA_PATH/$DATASET/test_labels.npy \
--train-labels $DATA_PATH/$DATASET/train_labels.npy
