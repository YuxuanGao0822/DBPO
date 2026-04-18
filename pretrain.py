"""Explicit DBP pretraining entrypoint."""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    workspace = hydra.utils.instantiate(cfg, _recursive_=False)
    workspace.run()


if __name__ == "__main__":
    main()
