import os
import re
import click
import numpy as np
from pathlib import Path
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from logzero import logger
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

from deepxml.data_utils import get_mlb, get_word_emb, output_res, build_vocab, convert_to_binary
from deepxml.dataset import MultiLabelDataset
from deepxml.models import Model, GPipeModel
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.meshprobenet import MeSHProbeNet, CorNetMeSHProbeNet
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN

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

def tokenize(sentence: str, sep='/SEP/'):
    return [token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('-i', '--input-file', type=click.Path(exists=True), help='Path of the input file containing texts to be evaluated.')
@click.option('-o', '--output-file', type=click.Path(), help='Path of the output file to save the results.')
@click.option('--vocab-path', type=click.Path(), default=None, help='Path of vocab, if it doesn\'t exit, build one and save it.')
@click.option('--w2v-model', type=click.Path(), default=None, help='Path of Gensim Word2Vec Model.')
@click.option('--vocab-size', type=click.INT, default=500000, help='Size of vocab.')
@click.option('--max-len', type=click.INT, default=500, help='Truncated length.')
def evaluate(data_cnf, model_cnf, input_file, output_file, vocab_path, w2v_model, vocab_size, max_len):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name, data_name = model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')
    
    logger.info('Loading Input Data')
    input_texts = np.loadtxt(input_file, dtype=str, delimiter='\n')
    logger.info(F'Size of Input Data: {len(input_texts)}')
    
    logger.info('Preprocessing Input Data')
    processed_texts = []
    for line in tqdm(input_texts, desc='Tokenizing'):
        processed_texts.append(' '.join(tokenize(line)))
    input_texts = np.array(processed_texts)
    
    logger.info('Loading Model')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    test_loader = DataLoader(MultiLabelDataset(input_texts), batch_size=1, num_workers=4)
    
    if not os.path.exists(vocab_path):
        logger.info(F'Building Vocab. {vocab_size}')
        vocab, emb_init = build_vocab(processed_texts, w2v_model, vocab_size=vocab_size)
        np.save(vocab_path, vocab)
        np.save(emb_path, emb_init)
    vocab = {word: i for i, word in enumerate(np.load(vocab_path))}
    logger.info(F'Vocab Size: {len(vocab)}')
    
    if 'gpipe' not in model_cnf:
        model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                      **data_cnf['model'], **model_cnf['model'])
    else:
        model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                           **data_cnf['model'], **model_cnf['model'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info('Predicting')
    scores, labels = [], []
    for data_x in tqdm(test_loader, desc='Predicting', leave=False):
        data_x = data_x.to(device)
        with torch.no_grad():
            output_scores, output_labels = torch.topk(model(data_x), k=model_cnf['predict'].get('k', 5))
        scores.append(output_scores.cpu())
        labels.append(output_labels.cpu())
    logger.info('Finish Predicting')
    labels = mlb.classes_[torch.cat(labels)]
    output_res(output_file, F'{model_name}-{data_name}', torch.cat(scores).numpy(), labels)

if __name__ == '__main__':
    evaluate()
