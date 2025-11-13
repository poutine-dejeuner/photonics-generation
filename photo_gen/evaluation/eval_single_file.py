import argparse
import yaml
import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from photo_gen.evaluation.evaluation import evaluate_model
from typing import Dict, Any


def eval_single_file(images: np.ndarray, savepath: Path = Path("."), fom: np.ndarray | None = None, **kwargs) -> Dict[str, Any]:
    evalconfigpath = "~/repos/diffusion-model/photo_gen/config/evaluation/config.yaml"
    evalconfigpath = Path(evalconfigpath).expanduser()
    # with open(evalconfigpath, 'r') as f:
    #     cfg = yaml.safe_load(f)
    with hydra.initialize_config_dir(config_dir=str(evalconfigpath.parent)):
        cfg = hydra.compose(config_name="config")

    OmegaConf.set_struct(cfg, False)

    cfg.model = {"name":"model"}
    # cfg = OmegaConf.create({"evaluation": cfg, "model": {"name": "model"}})
    evaluate_model(images, savepath, cfg, fom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".")
    args = parser.parse_args()
    path = Path(args.dir)
    images = np.load(path / "images.npy")
    fompath = path / "fom.npy"
    if fompath.exists():
        fom = np.load(fompath)
    else:
        fom = None
    eval_single_file(images=images, fom=fom)
