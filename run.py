import dotenv
from pytorch_lightning.cli import LightningCLI
import torch


if __name__ == '__main__':
    # load environment variables defined in .env
    dotenv.load_dotenv('.env')

    # register custom resolvers for omegaconf (registered on import)
    import src.utils.conf_resolvers

    # increase efficiency of matrix multiplies on GPUs with tensor cores
    torch.set_float32_matmul_precision('high')

    # launch the LightningCLI
    cli = LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={'overwrite': True},
    )