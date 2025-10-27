# template.py
import os

# --- Define project structure relative to current directory ---
structure = [
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "mlruns/.gitkeep",
    "data/.gitkeep",
    "results/.gitkeep",
    "notebooks/.gitkeep",
    "experiments/.gitkeep",
    "logs/.gitkeep",
    "configs/defaults.yaml",
    "configs/trainer/default.yaml",
    "configs/experiment/.gitkeep",
    "configs/dataset/.gitkeep",
    "configs/model/.gitkeep",
    "configs/optimizer/.gitkeep",
    "configs/scheduler/.gitkeep",
    "configs/grad/.gitkeep",
    "src/ags/__init__.py",
    "src/ags/registry.py",
    "src/ags/utils/__init__.py",
    "src/ags/data/__init__.py",
    "src/ags/data/base.py",
    "src/ags/models/__init__.py",
    "src/ags/models/resnet.py",
    "src/ags/optim/__init__.py",
    "src/ags/optim/optim_factory.py",
    "src/ags/grad/__init__.py",
    "src/ags/grad/base.py",
    "src/ags/grad/agc.py",
    "src/ags/grad/ags.py",
    "src/ags/trainer/__init__.py",
    "src/ags/trainer/loop.py",
    "src/ags/trainer/hooks.py",
    "src/ags/logger/__init__.py",
    "src/ags/logger/logger.py",
    "src/ags/entrypoints/__init__.py",
    "src/ags/entrypoints/train.py",
    "src/ags/entrypoints/eval.py",
    "tests/test_registry.py",
    "tests/test_grad_transforms.py",
    "tests/test_trainer_smoke.py",
]

# --- Create directories and empty files safely ---
for path in structure:
    dirname = os.path.dirname(path)
    if dirname:  # only make directory if not empty
        os.makedirs(dirname, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            pass  # create empty file
        print(f"✅ Created: {path}")
    else:
        print(f"⚠️ Skipped (exists): {path}")

print("\n🎉 Project structure generated successfully in current directory!")
print("Folders created:")
print("  src/ags/, configs/, experiments/, logs/, results/, notebooks/, tests/\n")
