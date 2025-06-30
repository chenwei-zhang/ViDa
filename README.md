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
conda create -n vida_env python=3.8 -y
conda activate vida_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install GSAE package
```bash
git clone https://github.com/KrishnaswamyLab/GSAE.git
cd GSAE
pip install -e . --no-deps
```

### 5. Install NUPACK
```unzip -q nupack-4.0.2.0.zip
pip install -U nupack -f nupack-4.0.2.0/package
```