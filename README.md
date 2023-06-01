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
Here's an example training ResNet-50 on CUB, using separate config files:
```bash
# "fit" tells Lightning to run the training loop
python run.py fit \
    -c src/configs/trainer/test_trainer.yaml \
    -c src/configs/model/resnet50.yaml \
    -c src/configs/data/cub.yaml
```

#### Backbones

We're using ResNet-50 and ViT-Base-16-224 as backbone feature extractors.
To standardize comparison, we use models pretrained on ImageNet-21k under the same settings,
available through [this repository](https://github.com/Alibaba-MIIL/ImageNet21K#1-pretrained-models--on-imagenet-21k-p-dataset).

**ResNet-50**: the weights need to be manually [downloaded](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth).

**ViT**: the weights can be downloaded and used through `timm` as `timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)`