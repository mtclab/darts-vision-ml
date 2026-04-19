import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_DIR = Path("weights")


def ensure_pretrained(name: str) -> str:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dest = WEIGHTS_DIR / name
    if dest.exists():
        return str(dest)

    model = YOLO(name)
    actual = Path(model.ckpt_path)

    if actual.exists() and actual.resolve() != dest.resolve():
        shutil.move(str(actual), str(dest))
    elif not dest.exists():
        cwd_copy = Path.cwd() / name
        if cwd_copy.exists() and cwd_copy.resolve() != dest.resolve():
            shutil.move(str(cwd_copy), str(dest))
        else:
            shutil.copy2(str(actual), str(dest))

    return str(dest)


def parse_device(device_str: str):
    if device_str.lower() == "cpu":
        return "cpu"
    parts = [int(x.strip()) for x in device_str.split(",")]
    return parts if len(parts) > 1 else parts[0]


def validate_dataset(config_path: str):
    path = Path(config_path).resolve()
    if not path.exists():
        print(f"[ERROR] Config not found: {path}")
        print("Run: python scripts/download_and_convert.py")
        return False

    import yaml
    with open(path) as f:
        ds_cfg = yaml.safe_load(f)
    ds_path = Path(ds_cfg["path"])
    if not ds_path.exists():
        print(f"[ERROR] Dataset path not found: {ds_path}")
        print(f"Config points to: {ds_cfg['path']}")
        print("Re-run inside Docker: python scripts/download_and_convert.py")
        return False

    return True