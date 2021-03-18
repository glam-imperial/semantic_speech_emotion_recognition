# Speech Emotion Recognition using Semantic Information

This repository provides training and evaluation code for the paper [`Speech Emotion Recognition using Semantic Information`](https://arxiv.org/pdf/2103.02993.pdf) (ICASSP 2021). If you use this codebase in your experiments please cite:

`Tzirakis, P., Nguyen, A., Zafeiriou, S., & Schuller, B. W. (2021). Speech Emotion Recognition using Semantic Information. arXiv preprint arXiv:2103.02993.`

This repository provides the following:

1. Word2Vec embeddings trained on the German SWC corpus.
2. Speech2Vec embeddings trained on the SEWA DB.
3. Code to train/evaluate word2vec and our unified model, i. e., paralinguistic and semantic feature extrators with a LSTM cell on top.

## Requirements
Below are listed the required modules to run the code.

  * aeneas
  * librosa
  * nltk
  * numpy
  * stop-words
  * tensorflow
  * torch
 
## Steps

1. Create the speech2vec segmentation by running `speech2word_mapping.py` in speech2vec folder.
2. Run `data_generator.py` to create tfrecords.
3. Run `train.py` to train the models, and `eval.py` to evaluate.
