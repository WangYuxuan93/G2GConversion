# Graph2Graph Conversion

This is the code we used in the following paper
>[Simple and Effective Graph-to-Graph Annotation Conversion](~)

>Yuxuan Wang*, Zhilin Lei*, Yuqiu Ji, Wanxiang Che

>Coling 2022


## Requirements

Python 3.6, PyTorch >=1.3.1, ...

## Running the experiments
First to the scripts folder:

    cd scripts

### Training
To train a g2g parser, simply run

    ./scripts/G2G-pattern.sh
Remeber to setup the paths for data, embeddings and other parameters.


## Thanks
Thanks to Ma et al. The implementation is based on the dependency parser by Ma
 et al. (2018) (https://github.com/XuezheMax/NeuroNLP2) and reuses part of its code.
 