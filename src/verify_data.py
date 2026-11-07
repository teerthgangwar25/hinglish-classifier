"""Checksum the four raw files so anyone (or any machine, e.g. Colab/Kaggle) can prove it
has the exact same data.   python src/verify_data.py --write   (once, then commit the json)
                           python src/verify_data.py           (verify)"""
import hashlib
import json
import sys

from preprocess import FILES, RAW, ROOT

OUT = ROOT / "data_checksums.json"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    current = {name: sha256(RAW / fname) for name, fname in FILES.items()}
    if "--write" in sys.argv:
        OUT.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print("wrote", OUT)
        return
    expected = json.loads(OUT.read_text(encoding="utf-8"))
    ok = True
    for name, digest in current.items():
        match = expected.get(name) == digest
        ok &= match
        print(f"{name:8s} {'OK' if match else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
