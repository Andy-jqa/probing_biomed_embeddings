# NER Probing Task

## Data
Preprocessed data is already in ```./data```

## 1. Cache the Contextualized Word Embeddings
First step is to cache the contextualized word embeddings. For this use:

```bash
python cache_dataset.py $EMBED_TYPE
# EMBED_TYPE is one of biomed_elmo, general_elmo or biomed_w2v
```

## 2. Train the model
After caching, you can train a model by:

```
$ python run.py --help
usage: run.py [-h] [--batch_size BATCH_SIZE] [--seed SEED] [--lr LR]
              [--num_epoch NUM_EPOCH] [--embed_type EMBED_TYPE]

Run probing experiments on bc2

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

## 3. Run the official evaluation codes
After training a model, test set predictions at the iteraction of best developement performance are located at ```${EMBED_TYPE}_seed${SEED}_log/predictions```.

Run the following script to evaluate the performance:

```bash
cd ./eavl_scripts
perl alt_eval.perl $PREDICTION_PATH
```
