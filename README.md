# Pyramid Pooling-based Local Profiles for Graph Classification

## The proposed approach

<img src="README/image-20230801203808547.png" alt="image-20230801203808547" style="zoom:80%;" />

In this paper, a pyramid pooling-based local profile (PPLP) approach is proposed. Node representations are sorted in a unique order, and then by performing pyramid pooling to these node representations, multi-scale features are extracted to obtain a fixed size graph level representation.

The proposed PPLP can be either employed as the readout module of a GNN, or work separately as a graph embedding algorithm using prepared node attributes.

The contributions of this paper are as follows: 

- PPLP  can be either embedded as the readout layer of a general GNN, or used separately as a graph embedding algorithm. Both schemes are experimentally investigated on performing graph classification tasks on ten representative benchmark datasets.
- Experimental results show the effectiveness of the proposed framework. The overall classification precision is higher when GNNs are embedded with PPLP as the readout layer, compared to the general global pooling-based readout layers. Also, as an independent graph embedding algorithm, PPLP is highly competitive to the state-of-the-art graph kernels and baseline GNNs.

## Dependencies
- networkx==2.8
- numpy==1.20.1
- Pillow==8.2.0
- scikit_learn==1.0.2
- scipy==1.6.2
- dgl==1.0.2
- pytorch==1.10.0

## Usage

### Install

` conda install --yes --file requirements.txt`

### run demo

```python
python main.py # evaluate model on TUDataset
```

## Cite

This paper has been accepted by [*IEEE International Conference on Systems, Man, and Cybernetics* (*SMC*) 2023](https://ieeesmc2023.org/).

Please cite our paper if you use this code in your research work.

```tex
The formal citation format will be updated after the paper is published.
```