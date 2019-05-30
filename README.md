# Probing Biomedical Embeddings from Language Models

Codes for probing tasks in [Probing Biomedical Embeddings from Language Models](https://arxiv.org/abs/1904.02181).

## Pre-trained Models
Please download BioELMo at https://github.com/Andy-jqa/bioelmo, and save the weights at ```./weights/biomed_elmo_weights.hdf5```, options at ```./weights/biomed_elmo_options.json``` and vocabulary at ```./dict/vocabulary.txt```.

For a general ELMo baseline, save the weights at ```./weights/general_elmo_weights.hdf5``` and options at ```./weights/general_elmo_options.json```.

## Probing Tasks
Probing task for NER on BC2GM is located in ```./ner_probing``` and probing task for NLI on MedNLI is located in ```./nli_probing```. Both directories contain detailed documentations.
