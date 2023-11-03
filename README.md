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
Python 3 : 
https://www.python.org/downloads/  

PyTorch 2.1.0 : 
For Mac / Linux:
$ pip install torch torchvision torchaudio 
For Windows:
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Dependencies
```
numpy==1.24.3
torch==2.1.0
plotly==5.17.0
matplotlib==3.5.1
tqdm==4.60.0
scikit-learn==1.3.1
networkx==3.1
phate==1.0.11
tensorboard==2.14
pandas==1.3.5
annoy==1.17.3
```

## Installation
```bash
# clone project   
git clone https://github.com/chenwei-zhang/ViDa

# install vida (recommond install in a virtual environment) 
conda activate <myenv>
cd ViDa
conda update pip -y
pip install -e .   
```   
**To install the Multistrand simulator, please follow [Multistrand officila tutorial](https://github.com/DNA-and-Natural-Algorithms-Group/multistrand).**

## Run ViDa in Docker
```bash
# clone project   
git clone https://github.com/chenwei-zhang/ViDa

# install vida iamge by Dockerfile
cd ViDa
docker build -t vida:v1.0.0 .

# create and run docker container from the built image
docker run -it --gpus all --rm vida:v1.0.0 
```   

## Data
The full data can be downloaded [here]().

```
data
  ├── raw_data
    ├── Hata-data
        ├── Hata-39.pkl.gz      # {'trajs_states', 'trajs_times', 'trajs_energies', 'trajs_types'}
    |── Gao-data
        ├── Gao-P0T0
           │-- Gao-P0T0-0.txt
           │-- Gao-P0T0-1.txt
           ...
        ├── Gao-P3T3
           │-- Gao-P3T3-0.txt
           │-- Gao-P3T3-1.txt
           ...
        ├── Gao-P3T3-hairpin
           │-- Gao-P3T3-hairpin-0.txt
           │-- Gao-P3T3-hairpin-1.txt
           ...
        ├── Gao-P4T4
           │-- Gao-P4T4-0.txt
           │-- Gao-P4T4-1.txt
           ...
        ├── Gao-P4T4-hairpin
           │-- Gao-P4T4-hairpin-0.txt
           │-- Gao-P4T4-hairpin-1.txt
           ...
  ├── post_data
    ├── Gao-P4T4       
        ├── preprocess_Gao-P4T4.npz         # {'dp_uniq', 'dp_og_uniq', 'pair_uniq', 'energy_uniq', 'trans_time', 'indices_uniq', 'indices_all'}				
        ├── time_Gao-P4T4.npz               # {'hold_time', 'hold_time_uniq', 'cum_time_uniq', 'freq_uniq', 'trj_id'}
        ├── adjmat_Gao-P4T4.npz             # {'adj_uniq'}				
        ├── scatt_Gao-P4T4.npz              # {'scar_uniq'}   NOTE: large file, download from Drive				
        ├── mpt-ged_Gao-P4T4.npz            # {'x_j', 'd_ij', 'e_ij', 'w_ij'}				
        ├── dataloader_Gao-P4T4.pkl.gz      # {'data_loader', 'train_loader', 'val_loader', 'dist_loader'}   NOTE: large file, download from Drive				
  ├── model_params
    ├── Hata-39_model.pt                # trained parameters for Hata-39
    ├── Gao-P0T0_model.pt               # trained parameters for Gao-P0T0
    ├── Gao-P3T3_model.pt               # trained parameters for Gao-P3T3
    ├── Gao-P3T3-hairpin_model.pt       # trained parameters for Gao-P3T3-hairpin
    ├── Gao-P4T4_model.pt               # trained parameters for Gao-P4T4
    ├── Gao-P4T4-hairpin_model.pt       # trained parameters for Gao-P4T4-hairpin
  ├── embed_data
  ├── config_template.json      # configuration template when training the model (change it accordingly)
  ...
  ...
```

## Workflow
### Load Multistrand's output data
    $ cd vida/data_processing

    $ python read_gaotxt.py --inpath /path/to/Gao-txtdata --rxn Gao-P4T4 --outpath /path/to/ouput
    (eg. $ python read_gaotxt.py --inpath ../../data/raw_data/Gao-data --rxn Gao-P4T4 --outpath ../../temp/Gao-P4T4.pkl.gz)

### Preprocess data
    $ python preprocess_data.py --inpath /path/to/pkl_file --outpath /path/to/output
    (eg. $ python preprocess_data.py  --inpath ../../temp/Gao-P4T4.pkl.gz  --outpath ../../temp/preprocess_Gao-P4T4.npz)

### Collect time data
    $ python comp_time.py --inpath /path/to/preprocess_data --outpath /path/to/output
    (eg. $ python  comp_time.py --inpath ../../temp/preprocess_Gao-P4T4.npz --outpath ../../temp/time_Gao-P4T4.npz)

### Convert dot-parenthesis to adjacency matrix
    $ python convert_adj.py --inpath /path/to/preprocess_data --output /path/to/output
    (eg. $ python convert_adj.py --inpath ../../temp/preprocess_Gao-P4T4.npz --outpath ../../temp/adjmat_Gao-P4T4.npz)

### Convert adjacency matrix to scattering coefficients
    $ python adj2scatt.py --inpath /path/to/adj_mat --num_strand <integer> --output /path/to/output
    (eg. $ python adj2scatt.py --inpath ../../temp/adjmat_Gao-P4T4.npz --num_strand 2 --outpath ../../temp/scatt_Gao-P4T4.npz)

### Compute MPF and GED distances
    $ cd vida/compute_distance

    $ python comp_dist.py --inpath /path/to/preprocess_data --holdtime /path/to/timedata --adjmat /path/to/adj_mat --outpath /path/to/output
    (eg. $ python comp_dist.py --inpath ../../temp/preprocess_Gao-P4T4.npz --holdtime ../../temp/time_Gao-P4T4.npz --adjmat ../../temp/adjmat_Gao-P4T4.npz --outpath ../../temp/mpt-ged_Gao-P4T4.npz)

### Create dataloader
    $ cd vida/models

    $ python dataloader.py --predata /path/to/preprocess_data --scatter /path/to/scatter_data --dist /path/to/mpt-ged_distdata --fconfig /path/to/config_file --outpath /path/to/output
    (eg. $ python dataloader.py --predata ../../temp/preprocess_Gao-P4T4.npz --scatter ../../temp/scatt_Gao-P4T4.npz --dist ../../temp/mpt-ged_Gao-P4T4.npz --fconfig ../../temp/config.json --outpath ../../temp/dataloader_Gao-P4T4.pkl.gz)

### Train the model
    $ python train_vida.py --data /path/to/dataloader --fconfig /path/to/config_file --outpath /path/to/output
    (eg. $ python train_vida.py --data ../../temp/dataloader_Gao-P4T4.pkl.gz --fconfig ../../temp/config.json --outpath ../../temp)

> If use Tensorbaod to monitor the training: <br>
> $ tensorboard --logdir=/path/to/model_config --port=6007

### Inference using the trained model and futher dimensionality reduction
    $ python embed_vida.py --data /path/to/dataloader --model /path/to/trained_model --fconfig /path/to/model_config_file --outpath /path/to/output
    (eg. $ python embed_vida.py --data ../../temp/dataloader_Gao-P4T4.pkl.gz --model ../../temp/model_config/23-1017-1729/model.pt --fconfig ../../temp/model_config/23-1017-1729/config.json --outpath ../../temp/model_config/23-1017-1729/embed_Gao-P4T4.npz)

### Interactive visualization
    $ python interact_plot.py --predata /path/to/preprocess_data --timedata /path/to/timedata  --embeddata /path/to/embeddata --outpath /path/to/output
    (eg. $ python interact_plot.py --predata ../../temp/preprocess_Gao-P4T4.npz --timedata ../../temp/time_Gao-P4T4.npz --embeddata ../../temp/embed_Gao-P4T4.npz --outpath ../../temp/plot_Gao-P4T4.html)


## Visualization results

> In short, what ViDa does is to biophysical-meaningfully reduce the dimensionality of Multistrand's outputs which are in very very high dimensions. Then visualize the features that are embedded in 2D space.

Here is a simple example of Multistrand's output for a DNA hybridization reaction with strand length of 9.
```bash
$ python run_multistrand.py
--------------------------------------------------------
GCGTTTCAC+GTGAAACGC
.(.......+.......).   t=0.000000 ms, dG=  0.16 kcal/mol
((.......+.......))   t=0.000060 ms, dG= -1.26 kcal/mol
GTGAAACGC+GCGTTTCAC
....(..((+))..)....   t=0.000526 ms, dG=  0.05 kcal/mol
....((.((+)).))....   t=0.000577 ms, dG= -1.46 kcal/mol
...(((.((+)).)))...   t=0.000608 ms, dG= -2.73 kcal/mol
...((((((+))))))...   t=0.000858 ms, dG= -7.90 kcal/mol
..(((((((+)))))))..   t=0.001025 ms, dG=-10.74 kcal/mol
.((((((((+)))))))).   t=0.001374 ms, dG= -9.79 kcal/mol
..(((((((+)))))))..   t=0.001421 ms, dG=-10.74 kcal/mol
.((((((((+)))))))).   t=0.002326 ms, dG= -9.79 kcal/mol
..(((((((+)))))))..   t=0.002601 ms, dG=-10.74 kcal/mol
..((((((.+.))))))..   t=0.002988 ms, dG= -9.33 kcal/mol
..(((((((+)))))))..   t=0.003122 ms, dG=-10.74 kcal/mol
..((((((.+.))))))..   t=0.003430 ms, dG= -9.33 kcal/mol
..(((((((+)))))))..   t=0.003570 ms, dG=-10.74 kcal/mol
..((((((.+.))))))..   t=0.003705 ms, dG= -9.33 kcal/mol
..(((((((+)))))))..   t=0.004507 ms, dG=-10.74 kcal/mol
..((((((.+.))))))..   t=0.006064 ms, dG= -9.33 kcal/mol
..(((((((+)))))))..   t=0.006210 ms, dG=-10.74 kcal/mol
..((((((.+.))))))..   t=0.006919 ms, dG= -9.33 kcal/mol
.(((((((.+.))))))).   t=0.007772 ms, dG= -8.37 kcal/mol
((((((((.+.))))))))   t=0.007780 ms, dG=-10.96 kcal/mol
(((((((((+)))))))))   t=0.008021 ms, dG=-12.38 kcal/mol
```

Here's the ViDa's visualization for the high-dimensional output features.
> **Visualization plots by ViDa for two different reactions**  
<div style="text-align:left">
    <img src="./assets/vis1.png" alt="cg" width="600"/>
</div>
<div style="text-align:left">
    <img src="./assets/vis2.png" alt="cg" width="600"/>
</div>

Here's the state-of-the-art coarse-grained (CG) method's visualization.
> **Visualization plots by CG for reaction Gao-P4T4**
<div style="text-align:left">
    <img src="./assets/cg.png" alt="cg" width="600"/>
</div>

> By comparison, we find that in the plot made by CG, each macrostate is an ensemble of secondary structures. However, with this scheme, structurally dissimilar secondary structures may be mapped to the same macrostate, making it difficult to interpret each macrostate and trajectories through them, and to distinguish between different reaction mechanisms. In contrast, ViDa's fine-grained embedding overcomes this limitation. ViDa's plots show distinct reaction trajectories, enabling users to interpret reaction mechanisms more straightforwardly and accurately. 

## Citation   
```
@inproceedings{Zhang-etal-2022,
  title={Visualizing {DNA} reaction trajectories with deep graph embedding approaches},
  author={Zhang, Chenwei and Dao Duc, Khanh and Condon, Anne},
  booktitle={Machine Learning for Structural Biology Workshop, NeurIPS},
  year={2022}
}
```   
