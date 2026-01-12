import copy
import yaml
from pathlib import Path
from src.ags.entrypoint.train import train


BASE_CFG = Path("configs/defaults.yaml")
TMP_CFG  = Path("configs/_tmp.yaml")


# -------------------------------
# Sweep definition
# -------------------------------
MODELS = ["cnn", "mlp", "vit"]
DATASETS = ["cifar10"]
BATCH_SIZES = [2, 4, 8]
LRS = [0.01, 0.1]   # SGD example
GC_METHODS = [None, "agc", "fisher_agc"]
SEEDS = [1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    base_cfg = load_yaml(BASE_CFG)

    for dataset in DATASETS:
        for model in MODELS:
            for bs in BATCH_SIZES:
                for lr in LRS:
                    for seed in SEEDS:
                        for gc in GC_METHODS:

                            cfg = copy.deepcopy(base_cfg)

                            # -----------------------
                            # Modify config
                            # -----------------------
                            cfg["dataset"] = dataset
                            cfg["model"] = model
                            cfg["seed"] = seed

                            # optimizer
                            cfg.setdefault("optimizer_cfg", {})
                            cfg["optimizer_cfg"]["lr"] = lr

                            # trainer
                            cfg.setdefault("trainer", {})
                            cfg["batch_size"] = bs

                            # MLflow naming
                            cfg["mlflow"]["run_name"] = (
                                f"{model}_{dataset}_lr{lr}_bs{bs}_{gc}_seed{seed}"
                            )
                            cfg["mlflow"]["tags"].update({
                                "model": model,
                                "dataset": dataset,
                                "lr": lr,
                                "batch_size": bs,
                                "gc": gc or "none",
                                "seed": seed,
                            })

                            # -----------------------
                            # Save temp config
                            # -----------------------
                            save_yaml(cfg, TMP_CFG)

                            print(f"\n▶ Running: {cfg['mlflow']['run_name']}")
                            train(str(TMP_CFG), gc)


if __name__ == "__main__":
    main()
