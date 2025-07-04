# ViDa

Code Review for the ViDa project.

## Installation

### 1. Clone the repository and checkout to the refactor branch
```bash
git clone git@github.com:chenwei-zhang/ViDa.git
cd ViDa
git checkout refactor
```

### 2. Create and activate a Conda environment
```bash
conda create -n vida_312 python=3.12 -y
conda activate vida_312
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install NUPACK
```bash
unzip -q nupack-4.0.2.0.zip
pip install -U nupack -f nupack-4.0.2.0/package
```

### 5. Install multistrand
```bash
git clone https://github.com/DNA-and-Natural-Algorithms-Group/multistrand
pip install -e multistrand
```

### 6. Install multigrain
```bash
git clone https://github.com/UBC-Mol-Prog/multigrain_private
pip install -e multigrain_private
```

### 7. Install GSAE package
```bash
git clone https://github.com/KrishnaswamyLab/GSAE.git
pip install -e GSAE --no-deps
```
