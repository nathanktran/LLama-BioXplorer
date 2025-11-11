import string
import os
import subprocess
import shutil
import yaml
import datetime
def clean_text(text):
    punctuation_to_remove = string.punctuation.replace('-', '').replace('_', '')
    cleaned_text = text.translate(str.maketrans('', '', punctuation_to_remove))
    cleaned_text = cleaned_text.lower()
    return cleaned_text


def generate_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def main(abstract_text, title):
    source_file1 = generate_unique_filename('input2_raw', 'txt')
    source_file2 = generate_unique_filename('input2', 'txt')


    a = f'{abstract_text} /SEP/ {title}'
    with open(source_file1, "w", encoding="utf-8") as text_file:
        text_file.write(a)


    a = f'{clean_text(abstract_text)} /SEP/ {clean_text(title)}'
    with open(source_file2, "w", encoding="utf-8") as text_file:
        text_file.write(a)


    os.environ['DATA_PATH'] = '/p/realai/BioXplorer/CorNet_Model/data'
    os.environ['CONFIG_PATH'] = '/p/realai/BioXplorer/CorNet_Model/configure/datasets'
    os.environ['DATASET'] = 'Mesh-2022-100K-12Terms-1'
    destination_dir = f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}"


    shutil.move(source_file1, destination_dir)
    shutil.move(source_file2, destination_dir)


    try:
        command = [
            'python', 'preprocess_website.py',
            '--text-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file1}",
            '--tokenized-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{source_file2}",
            '--vocab-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/vocab.npy"
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        return f"An error occurred during preprocessing: {e}"


    yaml_file_path = "/p/realai/BioXplorer/CorNet_Model/configure/datasets/Mesh-2022-100K-12Terms-1.yaml"
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)


    output_file2 = os.path.splitext(source_file2)[0] + '.npy'
    file_path_tokenised = f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/{output_file2}"
    yaml_content['input'] = {'texts': file_path_tokenised}


    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(yaml_content, file)


    try:
        command_predict = [
            'python', 'checkOutput3.py',
            '--data-cnf', f"{os.environ['CONFIG_PATH']}/Mesh-2022-100K-12Terms-1.yaml",
            '--model-cnf', "/p/realai/BioXplorer/CorNet_Model/configure/models/CorNetMeSHProbeNet-Mesh-2022-100K-12Terms-1.yaml",
            '--vocab-path', f"{os.environ['DATA_PATH']}/{os.environ['DATASET']}/vocab.npy"
        ]
        subprocess.run(command_predict, check=True)
        return "Prediction completed successfully"
    except Exception as e:
        return f"An error occurred during prediction: {e}"


if __name__ == "__main__":
    import sys
    abstract_text = sys.argv[1]
    title = sys.argv[2]
    result = main(abstract_text, title)
    print(result)