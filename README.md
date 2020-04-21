# SelfORE: Self-supervised Relational Feature Learning for Open Relation Extraction
The source code of paper "SelfORE: Self-supervised Relational Feature Learning for Open Relation Extraction"

**This is unrefactored version.** 

We use `pytorch-pretrained-bert 0.6.2` for pretrained BERT model and will migrate from `pytorch-pretrained-bert` to `transformers` in the future.

Adaptive clustering is inspired by [DEC](https://github.com/Deepayan137/DeepClustering).

Dataset:
* [NYT-FB](https://github.com/diegma/relation-autoencoder/blob/master/data-sample.txt)
* [TRex](https://hadyelsahar.github.io/t-rex/)

Run:
```sh
python main.py NUM_LABELS
```