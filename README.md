# Hard Problems in FGVC

## Setup

Clone the repository:
```bash
git clone https://github.com/catalys1/hard-problems-fgvc.git
cd hard-problems-fgvc
```

Create conda environment:
```bash
# recommend using mamba
conda install mamba
mamba create -n hp --file conda-env.txt

conda activate hp
```

### **Setup environment variables**

Run `initenv.py` to create a `.env` file with necessary environment variable definitions:
```bash
python initenv.py
# show the environment variables
cat .env
```
The environment variable `$WORKDIR` should be the path to the root of the repository (the same directory where `.env` lives).
The environment variable `$DATADIR` specifies a root directory where the code will look for each dataset.
By default this is `$WORKDIR/data`.
`$DATADIR` can be changed in `.env` to any path that contains the target datasets.

### **Install the `fgvcdata` package**

```bash
git clone https://github.com/catalys1/fgvc-data-pytorch.git
pip install -e fgvc-data-pytorch
```