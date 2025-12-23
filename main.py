from src.ags.entrypoint.train import train

def main():
    gc_methods = [
        None,
        "agc",
        "fisher_agc",
        "var_agc"
    ]
    for gc in gc_methods:
        train("configs/defaults.yaml", gc)

if __name__ == "__main__":
    main()