<div align="center">

# ViDa: Visualizing DNA hybridization trajectories with biophysics-informed deep graph embeddings

</div>

<!-- <a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/ViDa-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%7CMac%7CWindows-blue">
   <img src="https://img.shields.io/badge/language-python3-blue">
   <img src="https://img.shields.io/badge/gpus-Nvidia%7CApple%20M1-purple">
   <img src="https://img.shields.io/badge/licence-GNU-red">
</a>      <br> -->

![](https://img.shields.io/badge/ViDa-v1.0.0-green)
![](https://img.shields.io/badge/platform-Linux%7CMac%7CWindows-blue)
![](https://img.shields.io/badge/language-python3-blue)
![](https://img.shields.io/badge/gpus-Nvidia%7CApple%20M1-purple)
![](https://img.shields.io/badge/licence-GNU-yellow)
[![workshoppaper](https://img.shields.io/badge/paper-NeurIPS_MLSB-B90E0A)](https://www.mlsb.io/papers_2022/Visualizing_DNA_reaction_trajectories_with_deep_graph_embedding_approaches.pdf)

ViDa is a visualization tool for DNA reaction trajectories. 
It embeds DNA secondary structures emitted by elementary-step reaction simulators in a 2D landscape, using semi-supervised VAE embedding that leverages domain knowledge (free energy, minimum passage time, and graph edit distance) to determine custom training loss terms.
<br>
ViDa also supports interactive exploration of the landscape and trajectories.

Contact: Chenwei Zhang (cwzhang@cs.ubc.ca)


## About ViDa
Visualization tools can help synthetic biologists and molecular programmers understand the complex reactive pathways of nucleic acid reactions, which can be designed for many potential applications and can be modelled using a continuous-time Markov chain (CTMC). Here we present <b>ViDa</b>, a new visualization approach for DNA reaction trajectories that uses a 2D embedding of the secondary structure state space underlying the CTMC model. To this end, we integrate a scattering transform of the secondary structure adjacency, a variational autoencoder, and a nonlinear dimensionality reduction method. We augment the training loss with domain-specific supervised terms that capture both thermodynamic and kinetic features. We assess ViDa on two well-studied DNA hybridization reactions. Our results demonstrate that the domain-specific features lead to significant quality improvements over the state-of-the-art in DNA state space visualization, successfully separating different folding pathways and thus providing useful insights into dominant reaction mechanisms.

The framework of ViDa is shown below.

![vida_model](./assets/vida_model.png)   


## Pre-required software

```
Python 3 : https://www.python.org/downloads/  

PyTorch 2.0: 
For Mac: MPS acceleration is available on MacOS 12.3+
$ conda install pytorch::pytorch torchvision torchaudio -c pytorch 
For Linux / Windows:
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Dependencies
```
numpy==1.24.3
torch==2.1.0
plotly==5.17.0
matplotlib==3.5.1
PyYAML==6.0
tqdm==4.60.0
scikit-learn==1.3.1
networkx==3.1
phate==1.0.11
tensorboard==2.14
```

## Installation
```bash
# clone project   
git clone https://github.com/chenwei-zhang/ViDa

# install vida   
cd ViDa
pip install -e .   
pip install -r requirements.txt
```   

## Workflow

## 1. Loading data from Multistrand

## 2. Converting adjacency data to scattering coefficients

## 3. Create splits

## 4. Train the model



## Citation   
```
@inproceedings{Zhang-etal-2022,
  title={Visualizing {DNA} reaction trajectories with deep graph embedding approaches},
  author={Zhang, Chenwei and Dao Duc, Khanh and Condon, Anne},
  booktitle={Machine Learning for Structural Biology Workshop, NeurIPS},
  year={2022}
}
```   
