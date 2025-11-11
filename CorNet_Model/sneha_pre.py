import string
import os
import subprocess
import shutil


# abstract_text=["Background: To examine the relationship between chronic external and inter-nal head and neck lymphedema (HNL) and swallowing function in patientsfollowing head and neck cancer (HNC) treatment.Methods: Seventy-nine participants, 1-3 years post treatment were assessedfor external HNL using the MD Anderson Cancer Centre Lymphedema RatingScale, and internal HNL using Patterson's Radiotherapy Edema Rating Scale.Swallowing was assessed via instrumental, clinical and patient-reported out-come measures.Results: HNL presented as internal only (68%), combined external/internal(29%), and external only (1%). Laryngeal penetration/aspiration was confirmedin 20%. Stepwise multivariable regression models, that accounted for primarysite, revealed that a higher severity of external HNL and internal HNL wasassociated with more severe penetration/aspiration (P < .004 and P = .006,respectively), diet modification (P < .001 both), and poorer patient-reportedoutcomes (P = .037 and P = .014, respectively). Conclusion: Increased swallowing issues can be expected in patients pre-senting with more severe external HNL and/or internal HNL following HNCtreatment."]
abstract_text=["Ketamine has been found to have rapid and potent antidepressant activity. However, despite the ubiquitous brain expression of its molecular target, the N-methyl-d-aspartate receptor (NMDAR), it was not clear whether there is a selective, primary site for ketamine's antidepressant action. We found that ketamine injection in depressive-like mice specifically blocks NMDARs in lateral habenular (LHb) neurons, but not in hippocampal pyramidal neurons. This regional specificity depended on the use-dependent nature of ketamine as a channel blocker, local neural activity, and the extrasynaptic reservoir pool size of NMDARs. Activating hippocampal or inactivating LHb neurons swapped their ketamine sensitivity. Conditional knockout of NMDARs in the LHb occluded ketamine's antidepressant effects and blocked the systemic ketamine-induced elevation of serotonin and brain-derived neurotrophic factor in the hippocampus. This distinction of the primary versus secondary brain target(s) of ketamine should help with the design of more precise and efficient antidepressant treatments."]
# title=["Association between external and internal lymphedemaand chronic dysphagia following head and neck cancer treatment"]
title=["Brain region-specific action of ketamine as a rapid antidepressant"]


source_file1 = 'input_sneha_raw.txt'
source_file2 = 'input_sneha.txt'

a=f'{abstract_text[0]} /SEP/ {title[0]}'
print(a)

with open(source_file1, "w", encoding="utf-8") as text_file:
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

with open(source_file2, "w", encoding="utf-8") as text_file:
    text_file.write(a)




# Set environment variables
os.environ['DATA_PATH'] = '/p/realai/sneha/cornet2/CorNet/data'
os.environ['CONFIG_PATH'] = '/p/realai/sneha/cornet2/CorNet/configure/datasets'
os.environ['DATASET'] = 'Mesh-2022-100K'
destination_dir = f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}"



# Move the file
shutil.move(source_file1, destination_dir)
shutil.move(source_file2, destination_dir)

print(f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file1}")

if os.path.exists(f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file1}") and os.path.exists(f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file2}"):
    print("Input files moved to dataset folder")
else:
    print("Input files are missing")

try:
# Construct the command
    command = [
        'python', 'preprocess.py',
        '--text-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file1}",
        '--tokenized-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file2}",
        '--vocab-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/vocab.npy"
    ]

    # Run the command
    subprocess.run(command, check=True)

    # Do a check to see if npy files are created
    print("INput texts are preprocessed successfully")
except Exception as e:
    print(f"An unexpected error occurred: {e}")




# Add file path to yaml file
import yaml

# Define the path to the YAML file
yaml_file_path = "/p/realai/sneha/cornet2/CorNet/configure/datasets/Mesh-2022-100K.yaml"


# Load the existing YAML file
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.safe_load(file)

output_file2 = os.path.splitext(source_file2)[0] + '.npy'
file_path_tokenised= f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{output_file2}"
print(output_file2)
# Add the input section with the specified path
yaml_content['input'] = {
    'texts': file_path_tokenised
}

# Write the updated content back to the YAML file
with open(yaml_file_path, 'w') as file:
    yaml.safe_dump(yaml_content, file)

print("YAML file updated successfully!")


# Predicting

command_predict = [
        'python', 'checkOutput3.py',
        '--data-cnf', f"{os.environ['CONFIG_PATH']}/Mesh-2022-100K.yaml",
        '--model-cnf', "/p/realai/sneha/cornet2/CorNet/configure/models/CorNetMeSHProbeNet-Mesh-2022-100K.yaml",
        '--vocab-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/vocab.npy"
    ]

subprocess.run(command_predict, check=True)

# Delete the source files and the tokenized output file
try:
    os.remove(f"{destination_dir}/{source_file1}")
    os.remove(f"{destination_dir}/{source_file2}")
    os.remove(file_path_tokenized)
    print("Cleanup complete! Source files and output file deleted.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred during cleanup: {e}")


# python checkOutput3.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --vocab-path $DATA_PATH/$DATASET/vocab.npy
