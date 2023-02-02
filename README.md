# Layer Adaptive Graph Neural Networks(LAGNN)

code for LAGNN

##  Environment Settings

* python == 3.8.10
* torch == 1.10.0

## Parameter Settings

- middle_layer_num: number of hidden units
- value_layer_dropout: probility of layer_dropout (keep)
- temperature: temperature in gumbel softmax
- epoch: number of epochs to train the base model
- stage1_epoch: Number of epochs to pre-train
- seed: random seed
- lr: learning rate
- weight_decay: weight decay (L2 loss on parameters)
- hidden: embedding dimension
- dropout: dropout rate
- dataset: dataset cora, citeseer, pubmed, coauthor-cs, coauthor-phy,  amazon-com
- epoch: number of epochs to train the base model
- stage1_epoch: Number of epochs to pre-train
- weight_choose_share: share choose weight between layer
- choose_weight_type: 0 for input, 1 for input and output, 2 for resnet, 3 for random dropout layer
- choose_weight_layernum: the number of weight layer
- linear_decay: whether probability decays linearly with the number of layers or not

## Basic Usage

~~~
python ./LAGNN/train.py 
~~~