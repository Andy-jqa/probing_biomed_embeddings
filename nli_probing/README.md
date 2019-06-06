# NLI Probing Task

## Data
Please download the MedNLI dataset [here](https://physionet.org/physiotools/mimic-code/mednli/) (you might need to apply for a license to download it). Put the downloaded jsonline files in ```./data```, and run the preprocessing code:

```bash
python preprocess.py
```

## 1. Cache the Contextualized Word Embeddings
First step is to cache the contextualized word embeddings. For this use:

```bash
python cache_dataset.py --embed_type $EMBED_TYPE
# EMBED_TYPE is one of biomed_elmo, general_elmo or biomed_w2v
```

## 2. Train the model
After caching, you can train a model by:

```
$ python run.py --help
usage: run.py [-h] [--batch_size BATCH_SIZE] [--seed SEED] [--lr LR]
              [--num_epoch NUM_EPOCH] [--embed_type EMBED_TYPE]

Run probing experiments on mednli

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size (Default: 32)
  --seed SEED           random seed (Default: 0)
  --lr LR               Adam learning rate (Default: 0.002)
  --num_epoch NUM_EPOCH
                        number of epochs to train (Default: 10)
  --embed_type EMBED_TYPE
                        Cache the specified embedding type of the dataset.
                        Possible types: "biomed_elmo", "biomed_w2v",
                        "general_elmo"

```

Logs will be saved at ```${EMBED_TYPE}_seed${SEED}_log```.
