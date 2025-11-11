#!/usr/bin/env bash

DATA_PATH=/p/realai/sneha/cornet2/CorNet/data

# DATASET=EUR-Lex
# DATASET=AmazonCat-13K
# DATASET=Mesh-2022
DATASET=Mesh-2022-100K
# DATASET=Mesh-2022-pubMed
#DATASET=Wiki-500K

#MODEL=XMLCNN
#MODEL=CorNetXMLCNN
#MODEL=BertXML
#MODEL=CorNetBertXML
#MODEL=MeSHProbeNet
MODEL=CorNetMeSHProbeNet
#MODEL=AttentionXML
#MODEL=CorNetAttentionXML

python checkOutput3.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --vocab-path $DATA_PATH/$DATASET/vocab.npy

