"""
Download leonli66/latent-cot-finewebedu parquet shards to ./data/parquet/.

Usage:
  uv run python scripts/download_data.py                      # all 8192 shards
  uv run python scripts/download_data.py --max_shards 64      # first 64 shards
  uv run python scripts/download_data.py --workers 32         # parallelism
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

load_dotenv()

REPO_ID = "leonli66/latent-cot-finewebedu"
DEFAULT_OUT = Path("data/parquet")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--max_shards", type=int, default=None, help="Limit number of shards (default: all)")
    p.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    return p.parse_args()


def download_shard(filename: str, out_dir: Path, token: str) -> Path:
    dest = out_dir / filename
    if dest.exists():
        return dest  # already downloaded

    tmp = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=token,
        cache_dir=None,           # don't use HF cache — write directly
        local_dir=str(out_dir),   # writes to out_dir/filename
    )
    return Path(tmp)


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token or token == "hf_...":
        sys.exit("ERROR: HF_TOKEN not set. Fill in .env first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # List all parquet shards
    api = HfApi(token=token)
    all_files = [
        f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    all_files.sort()

    if args.max_shards is not None:
        all_files = all_files[: args.max_shards]

    already = sum(1 for f in all_files if (out_dir / f).exists())
    print(f"Shards: {len(all_files)} total, {already} already downloaded, {len(all_files) - already} to fetch")
    print(f"Output: {out_dir.resolve()}")
    print(f"Workers: {args.workers}")

    todo = [f for f in all_files if not (out_dir / f).exists()]
    if not todo:
        print("All shards already present.")
        return

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_shard, f, out_dir, token): f for f in todo}
        with tqdm(total=len(todo), unit="shard") as bar:
            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((fname, str(e)))
                    tqdm.write(f"ERROR {fname}: {e}")
                bar.update(1)

    print(f"\nDone. {len(todo) - len(errors)} downloaded, {len(errors)} failed.")
    if errors:
        print("Failed shards:")
        for f, e in errors:
            print(f"  {f}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
