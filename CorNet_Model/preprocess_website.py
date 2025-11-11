import os
import re
import click
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from logzero import logger

from deepxml.data_utils import build_vocab, convert_to_binary


def tokenize(sentence: str, sep='/SEP/'):
    # We added a /SEP/ symbol between titles and descriptions such as Amazon datasets.
    return [token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]


@click.command()
@click.option('--text-path', type=click.Path(exists=True), help='Path of text.')
@click.option('--tokenized-path', type=click.Path(), default=None, help='Path of tokenized text.')
@click.option('--label-path', type=click.Path(exists=True), default=None, help='Path of labels.')
@click.option('--vocab-path', type=click.Path(), default=None,
              help='Path of vocab, if it doesn\'t exit, build one and save it.')
@click.option('--emb-path', type=click.Path(), default='/p/realai/sneha/cornet2/CorNet/data/Mesh-2022-100K-12Terms-1/emb_init.npy', help='Path of word embedding.')
@click.option('--w2v-model', type=click.Path(), default=None, help='Path of Gensim Word2Vec Model.')
@click.option('--vocab-size', type=click.INT, default=500000, help='Size of vocab.')
@click.option('--max-len', type=click.INT, default=500, help='Truncated length.')
# def main(text_path, tokenized_path, label_path, vocab_path, emb_path, w2v_model, vocab_size, max_len):
#     if tokenized_path is not None:
#         logger.info(F'Tokenizing Text. {text_path}')
#         with open(text_path) as fp, open(tokenized_path, 'w') as fout:
#             for line in tqdm(fp, desc='Tokenizing'):
#                 print(*tokenize(line), file=fout)
#         text_path = tokenized_path

#     if not os.path.exists(vocab_path):
#         logger.info(F'Building Vocab. {text_path}')
#         logger.info(F'Building Vocab. {vocab_size}')
#         with open(text_path) as fp:
#             vocab, emb_init = build_vocab(fp, w2v_model, vocab_size=vocab_size)
#         np.save(vocab_path, vocab)
#         np.save(emb_path, emb_init)
#     vocab = {word: i for i, word in enumerate(np.load(vocab_path))}
#     logger.info(F'Vocab Size: {len(vocab)}')

#     logger.info(F'Getting Dataset: {text_path} Max Length: {max_len}')
#     texts, labels = convert_to_binary(text_path, label_path, max_len, vocab)
#     logger.info(F'Size of Samples: {len(texts)}')
#     print(text_path)
#     np.save(os.path.splitext(text_path)[0], texts)
#     if labels is not None:
#         assert len(texts) == len(labels)
#         np.save(os.path.splitext(label_path)[0], labels)

def main(text_path, tokenized_path, label_path, vocab_path, emb_path, w2v_model, vocab_size, max_len):
    if tokenized_path is not None:
        logger.info(F'Tokenizing Text. {text_path}')
        with open(text_path) as fp, open(tokenized_path, 'w') as fout:
            for line in tqdm(fp, desc='Tokenizing'):
                print(*tokenize(line), file=fout)
        text_path = tokenized_path

    if not os.path.exists(vocab_path):
        logger.info(F'Building Vocab. {text_path}')
        logger.info(F'Building Vocab. {vocab_size}')
        with open(text_path) as fp:
            vocab, emb_init = build_vocab(fp, w2v_model, vocab_size=vocab_size)
        np.save(vocab_path, vocab)
        np.save(emb_path, emb_init)
    vocab = {word: i for i, word in enumerate(np.load(vocab_path))}
    logger.info(F'Vocab Size: {len(vocab)}')

    logger.info(F'Getting Dataset: {text_path} Max Length: {max_len}')
    texts, labels = convert_to_binary(text_path, label_path, max_len, vocab)
    
    logger.info(F'Size of Samples: {len(texts)}')
    
    text_output_path = os.path.splitext(text_path)[0] + '.npy'
    label_output_path = os.path.splitext(label_path)[0] + '.npy' if label_path else None
    
    if len(texts) > 0:
        logger.info(F'Saving texts to {text_output_path}')
        np.save(text_output_path, texts)
    else:
        logger.error('No texts to save.')

    if labels is not None:
        if len(texts) == len(labels):
            logger.info(F'Saving labels to {label_output_path}')
            np.save(label_output_path, labels)
        else:
            logger.error('Mismatch between texts and labels.')
    else:
        logger.info('No labels to save.')

if __name__ == '__main__':
    main()
