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
mamba env create -f env.yaml

conda activate hp
```

#### Setup environment variables

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

#### Install the `fgvcdata` package

```bash
git clone https://github.com/catalys1/fgvc-data-pytorch.git
pip install -e fgvc-data-pytorch
```


## Development

#### Run the training script

The entrypoint is `run.py`.
We need to specify at least one configuration file (containing configurations for the trainer, model, and data).
We can also combine different configuration files.
Here's an example of training ResNet-50 on CUB.
```bash
# create a single config file by merging configs for trainer, data, and model
python utils/configs.py src/configs/trainer/test_trainer.yaml src/configs/data/cub.yaml src/configs/models/resnet50.yaml -f config.yaml

# "fit" tells Lightning to run the training loop
python run.py fit -c config.yaml
```

#### Backbones

We're using ResNet-50 and ViT-Base-16-224 as backbone feature extractors.
To standardize comparison, we use models pretrained on ImageNet-1k with supervised learning.

**ResNet-50**: we use the weights available from `torchvision` as `IMAGENET1K_V2`,
which we found to give more stable performance than the default weights from `timm`.

**ViT**: We use the weights available from `timm` as `vit_base_patch16_224.augreg_in1k`.