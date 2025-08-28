#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Set


def find_wavs(root_dir: Path, sub_dir_name: str) -> Dict[str, str]:
    """Scan for wav files under a subdirectory and map index -> absolute path.

    Index is derived from filename without extension (e.g., p226_001 from p226_001.wav).
    """
    target_dir = root_dir / sub_dir_name
    wav_map: Dict[str, str] = {}
    if not target_dir.exists():
        return wav_map

    for dirpath, _dirnames, filenames in os.walk(target_dir):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue
            idx = os.path.splitext(fname)[0]
            abs_path = str(Path(dirpath, fname).resolve())
            wav_map[idx] = abs_path
    return wav_map


def read_indices_from_list(file_path: Path) -> List[str]:
    """Read indices from a txt that may be in different formats.

    Supported line formats:
    - index|...
    - index (single token)
    - .../p226_001.wav (derive index from basename)
    """
    indices: List[str] = []
    if not file_path.exists():
        return indices

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Prefer the first pipe-separated token as index
            if "|" in line:
                token = line.split("|", 1)[0].strip()
                if token:
                    indices.append(token)
                    continue
            # If it's a path, derive from basename
            if "/" in line or "\\" in line:
                base = os.path.basename(line)
                token = os.path.splitext(base)[0]
                if token:
                    indices.append(token)
                    continue
            # Fallback: whole line is the index
            indices.append(line)
    return indices


 


def write_list(out_path: Path, ordered_indices: List[str], noisy_map: Dict[str, str], clean_map: Dict[str, str]) -> None:
    """Write lines in the format index|noisy_path|clean_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for idx in ordered_indices:
            noisy = noisy_map[idx]
            clean = clean_map[idx]
            fo.write(f"{idx}|{noisy}|{clean}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse VoiceBank+DEMAND file list and create train/test txts.")
    parser.add_argument("--path", required=True, help="Dataset root containing wav_clean/ and wav_noisy/")
    parser.add_argument("--clean_dir", default="wav_clean", help="Subdirectory name for clean wavs")
    parser.add_argument("--noisy_dir", default="wav_noisy", help="Subdirectory name for noisy wavs")
    parser.add_argument("--train_list", default="training.txt", help="Optional existing list to determine train indices")
    parser.add_argument("--test_list", default="test.txt", help="Optional existing list to determine test indices")
    parser.add_argument("--train_out", default="training_list.txt", help="Output training list path (written under --path if relative)")
    parser.add_argument("--test_out", default="test_list.txt", help="Output test list path (written under --path if relative)")

    args = parser.parse_args()

    root = Path(args.path).resolve()
    clean_map = find_wavs(root, args.clean_dir)
    noisy_map = find_wavs(root, args.noisy_dir)

    # Intersect indices available in both maps
    common_indices: Set[str] = set(clean_map.keys()) & set(noisy_map.keys())

    if not common_indices:
        raise SystemExit("No paired clean/noisy files found. Check directory structure and extension.")

    # Determine train/test split
    train_indices: Set[str] = set()
    test_indices: Set[str] = set()

    train_list_path = Path(args.train_list)
    test_list_path = Path(args.test_list)

    # If relative, treat as inside dataset root
    if not train_list_path.is_absolute():
        train_list_path = root / train_list_path
    if not test_list_path.is_absolute():
        test_list_path = root / test_list_path

    train_from_file = read_indices_from_list(train_list_path)
    test_from_file = read_indices_from_list(test_list_path)

    if train_from_file or test_from_file:
        if train_from_file:
            train_indices = set(train_from_file) & common_indices
        if test_from_file:
            test_indices = set(test_from_file) & common_indices
        # If only one side provided, infer the other
        if train_from_file and not test_from_file:
            test_indices = common_indices - train_indices
        if test_from_file and not train_from_file:
            train_indices = common_indices - test_indices
    else:
        # If nothing provided, put everything into training
        train_indices = set(common_indices)
        test_indices = set()

    # Sort indices for stability
    train_sorted = sorted(train_indices)
    test_sorted = sorted(test_indices)

    # Resolve output paths
    train_out = Path(args.train_out)
    test_out = Path(args.test_out)
    if not train_out.is_absolute():
        train_out = root / train_out
    if not test_out.is_absolute():
        test_out = root / test_out

    if train_sorted:
        write_list(train_out, train_sorted, noisy_map, clean_map)
    if test_sorted:
        write_list(test_out, test_sorted, noisy_map, clean_map)

    # Brief stdout summary
    print(f"Paired files: {len(common_indices)} | Train: {len(train_sorted)} | Test: {len(test_sorted)}")
    print(f"Train list written to: {train_out}")
    if test_sorted:
        print(f"Test list written to: {test_out}")


if __name__ == "__main__":
    main()
