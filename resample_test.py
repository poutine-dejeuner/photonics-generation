import os
from tqdm import tqdm
import hydra
from icecream import ic
from omegaconf import OmegaConf

from train3 import inference
from nanophoto.evaluation.evalgen import eval_metrics


@hydra.main(config_path="config", config_name="config")
def resample_test(cfg):
    cfg.checkpoint_load_path = os.path.join(os.environ["SCRATCH"],
                                            "nanophoto/diffusion/train3/7121883")
    trainset_path = "~/scratch/nanophoto/evalgen/topoptim/data"
    trainset_path = os.path.expanduser(trainset_path)
    n_resamples = 3
    chkpt_path = os.path.expanduser(cfg.checkpoint_load_path)
    chkpt_path = os.path.join(chkpt_path, "checkpoint.pt")
    chkpt_dir = os.path.dirname(chkpt_path)
    for i in tqdm(range(n_resamples)):
        savepath = os.path.join(chkpt_dir, f"test{i}")
        os.makedirs(savepath, exist_ok=True)
        images, fom = inference(cfg=cfg, checkpoint_path=chkpt_path,
                                savepath=savepath, meep_eval=False)
        eval_conf = [OmegaConf.create({"name": f"test{i}", "path": chkpt_dir})]
        eval_metrics(datasets=eval_conf,
                     trainset_path=trainset_path, n_samples=256)


if __name__ == "__main__":
    resample_test()
