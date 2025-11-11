import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logzero import logger

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model, GPipeModel
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.meshprobenet import MeSHProbeNet, CorNetMeSHProbeNet
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN

import torch
import gc

def load_vocab(vocab_path):
    # Load the vocabulary from the .npy file
    vocab_array = np.load(vocab_path, allow_pickle=True)
    vocab = {word: i for i, word in enumerate(vocab_array)}
    return vocab

def create_reverse_vocab(vocab):
    # Create a reverse mapping from indices to words
    reverse_vocab = {index: word for word, index in vocab.items()}
    return reverse_vocab

def indices_to_words(indices, reverse_vocab):
    # Convert indices to words
    words = [reverse_vocab.get(index, '<UNK>') for index in indices]
    return words

# Example usage

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


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('--vocab-path', type=click.Path(), default=None,
              help='Path of vocab, if it doesn\'t exit, build one and save it.')
def main(data_cnf, model_cnf, mode, vocab_path):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

    if mode is None or mode == 'eval':
        logger.info('Loading Test Set')
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['input']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        logger.info('Predicting')
        # Clear cache before model loading
        
        # test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'], num_workers=4)
        test_loader = DataLoader(MultiLabelDataset(test_x),batch_size=1, num_workers=4)
        if 'gpipe' not in model_cnf:
            if model is None:
                model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                              **data_cnf['model'], **model_cnf['model'])
        else:
            if model is None:
                model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init, 
                                   **data_cnf['model'], **model_cnf['model'])
        # Move data to the specified device
        # test_loader.dataset.data = test_loader.dataset.data.to(device)


        # scores, labels = model.predict(test_loader, k=5)
        scores, labels = model.predict(test_loader, k=12)
        # print(labels)
        # scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
        logger.info('Finish Predicting')
        # labels = mlb.classes_[labels]
        
   

        vocab = load_vocab(vocab_path)
        reverse_vocab = create_reverse_vocab(vocab)

        for label_set in labels:
            words = indices_to_words(label_set, reverse_vocab)
            print("DEBUG: ")
            print(words)

        # Assuming label_indices is a 2D list or numpy array
        # for label_set in labels:
        #     words = indices_to_words(label_set, reverse_vocab)
        #     print(words)

        # output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', scores, labels)


if __name__ == '__main__':
    main()
