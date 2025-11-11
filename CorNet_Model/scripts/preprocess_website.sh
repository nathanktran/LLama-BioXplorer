import string
import os
import subprocess

abstract_text=["Background: To examine the relationship between chronic external and inter-nal head and neck lymphedema (HNL) and swallowing function in patientsfollowing head and neck cancer (HNC) treatment.Methods: Seventy-nine participants, 1-3 years post treatment were assessedfor external HNL using the MD Anderson Cancer Centre Lymphedema RatingScale, and internal HNL using Patterson's Radiotherapy Edema Rating Scale.Swallowing was assessed via instrumental, clinical and patient-reported out-come measures.Results: HNL presented as internal only (68%), combined external/internal(29%), and external only (1%). Laryngeal penetration/aspiration was confirmedin 20%. Stepwise multivariable regression models, that accounted for primarysite, revealed that a higher severity of external HNL and internal HNL wasassociated with more severe penetration/aspiration (P < .004 and P = .006,respectively), diet modification (P < .001 both), and poorer patient-reportedoutcomes (P = .037 and P = .014, respectively). Conclusion: Increased swallowing issues can be expected in patients pre-senting with more severe external HNL and/or internal HNL following HNCtreatment."]
title=["Association between external and internal lymphedemaand chronic dysphagia following head and neck cancer treatment"]

a=f'{abstract_text[0]} /SEP/ {title[0]}'
print(a)

with open("input2_raw.txt", "w", encoding="utf-8") as text_file:
    text_file.write(a)



def clean_text(text):
    # Define punctuation to remove, excluding hyphen and underscore
    punctuation_to_remove = string.punctuation.replace('-', '').replace('_', '')
    
    # Remove specified punctuation
    cleaned_text = text.translate(str.maketrans('', '', punctuation_to_remove))
    
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    return cleaned_text

# print(clean_text(abstract_text[0]) + "/SEP" + clean_text(title[0]))

a=f'{clean_text(abstract_text[0])} /SEP/ {clean_text(title[0])}'
print(a)

with open("input2.txt", "w", encoding="utf-8") as text_file:
    text_file.write(a)


# DATA_PATH=/p/realai/sneha/cornet2/CorNet/data
# DATASET=Mesh-2022-100K



# python preprocess.py \
# --text-path $DATA_PATH/$DATASET/input_raw.txt \
# --tokenized-path $DATA_PATH/$DATASET/input.txt \
# --vocab-path $DATA_PATH/$DATASET/vocab.npy

