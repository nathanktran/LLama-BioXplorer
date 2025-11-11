import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from logzero import logger

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model, GPipeModel
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.meshprobenet import MeSHProbeNet, CorNetMeSHProbeNet
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from torch.utils.data import Dataset

model_dict = {
        'AttentionXML': AttentionXML,
        'CorNetAttentionXML': CorNetAttentionXML,
        'MeSHProbeNet': MeSHProbeNet,
        'CorNetMeSHProbeNet': CorNetMeSHProbeNet,
        'BertXML': BertXML,
        'CorNetBertXML': CorNetBertXML,
        'XMLCNN': XMLCNN,
        'CorNetXMLCNN': CorNetXMLCNN
        }
class MultiLabelDatasetSingle(Dataset):
    def __init__(self, data_x):
        self.data_x = data_x

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx]

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('-i', '--input-file', type=click.Path(exists=True), help='Path of the input file containing texts to be evaluated.')
@click.option('-o', '--output-file', type=click.Path(), help='Path of the output file to save the results.')
def evaluate(data_cnf, model_cnf, input_file, output_file):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name, data_name = model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')
    
    logger.info('Loading Input Data')
    input_texts = np.loadtxt(input_file, dtype=str, delimiter='\n').tolist()
    if isinstance(input_texts, str):  # When there's only one line, np.loadtxt returns a string
        input_texts = [input_texts]
    logger.info(F'Size of Input Data: {len(input_texts)}')
    # logger.info(F'Size of Input Data: {len(input_texts)}')
    
    logger.info('Loading Model')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    # test_loader = DataLoader(MultiLabelDataset(input_texts), model_cnf['predict']['batch_size'], num_workers=4)
    test_loader = DataLoader(MultiLabelDataset(input_texts), batch_size=1, num_workers=4)
    
    
    if 'gpipe' not in model_cnf:
        model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                      **data_cnf['model'], **model_cnf['model'])
    else:
        model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                           **data_cnf['model'], **model_cnf['model'])
    
    logger.info('Predicting')
    # scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
    scores, labels = model.predict(test_loader, k=5)
    logger.info('Finish Predicting')
    labels = mlb.classes_[labels]
    output_res(output_file, F'{model_name}-{data_name}', scores, labels)

if __name__ == '__main__':
    evaluate()