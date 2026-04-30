from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lossless_caps import (  # noqa: E402
    LOSSLESS_CAPS_CASEOPS_V1,
    encode_lossless_caps_v2,
    surface_piece_original_byte_counts,
)

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_TOKENS = 10_000_000
BOS_ID = 1


def _write_shard(out_path: pathlib.Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _iter_docs(path: str):
    if path == "-":
        fh = sys.stdin
        close = False
    else:
        fh = open(path, "r", encoding="utf-8")
        close = True
    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj
    finally:
        if close:
            fh.close()


def _token_original_byte_counts(sp: spm.SentencePieceProcessor, transformed_text: str) -> np.ndarray:
    proto = sp.encode_as_immutable_proto(transformed_text)
    byte_counts = surface_piece_original_byte_counts(
        (piece.surface for piece in proto.pieces),
        text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
    )
    return np.asarray(list(byte_counts), dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="docs_selected.jsonl path, or '-' for stdin")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--sp", required=True, type=pathlib.Path)
    ap.add_argument("--val-docs", type=int, default=50_000)
    ap.add_argument("--max-train-shards", type=int, default=0)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.sp))
    print(f"loaded sp: vocab={sp.vocab_size()}", flush=True)

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    val_buf_tokens: list[int] = []
    val_buf_bytes: list[int] = []
    train_buf: list[int] = []
    val_written = 0
    train_written = 0
    n_docs = 0

    for text in _iter_docs(args.docs):
        transformed = encode_lossless_caps_v2(text)
        token_ids = [BOS_ID] + sp.encode(transformed, out_type=int)
        if n_docs < args.val_docs:
            byte_counts = _token_original_byte_counts(sp, transformed)
            val_buf_tokens.extend(token_ids)
            val_buf_bytes.append(0)
            val_buf_bytes.extend(int(b) for b in byte_counts)
            while len(val_buf_tokens) >= SHARD_TOKENS:
                _write_shard(
                    train_out / f"fineweb_val_{val_written:06d}.bin",
                    np.array(val_buf_tokens[:SHARD_TOKENS], dtype=np.uint16),
                )
                _write_shard(
                    train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                    np.array(val_buf_bytes[:SHARD_TOKENS], dtype=np.uint16),
                )
                val_buf_tokens = val_buf_tokens[SHARD_TOKENS:]
                val_buf_bytes = val_buf_bytes[SHARD_TOKENS:]
                val_written += 1
        else:
            train_buf.extend(token_ids)
            while len(train_buf) >= SHARD_TOKENS:
                _write_shard(
                    train_out / f"fineweb_train_{train_written:06d}.bin",
                    np.array(train_buf[:SHARD_TOKENS], dtype=np.uint16),
                )
                train_buf = train_buf[SHARD_TOKENS:]
                train_written += 1
                if args.max_train_shards and train_written >= args.max_train_shards:
                    if val_buf_tokens:
                        _write_shard(
                            train_out / f"fineweb_val_{val_written:06d}.bin",
                            np.array(val_buf_tokens, dtype=np.uint16),
                        )
                        _write_shard(
                            train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                            np.array(val_buf_bytes, dtype=np.uint16),
                        )
                        val_written += 1
                        val_buf_tokens = []
                        val_buf_bytes = []
                    print(
                        f"done docs={n_docs + 1} train_shards={train_written} val_shards={val_written}",
                        flush=True,
                    )
                    return
        n_docs += 1
        if n_docs % 10_000 == 0:
            print(
                f"processed {n_docs} docs train_shards={train_written} val_shards={val_written}",
                flush=True,
            )

    if val_buf_tokens:
        _write_shard(
            train_out / f"fineweb_val_{val_written:06d}.bin",
            np.array(val_buf_tokens, dtype=np.uint16),
        )
        _write_shard(
            train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
            np.array(val_buf_bytes, dtype=np.uint16),
        )
    if train_buf:
        _write_shard(
            train_out / f"fineweb_train_{train_written:06d}.bin",
            np.array(train_buf, dtype=np.uint16),
        )
    print(
        f"done docs={n_docs} train_shards={train_written + (1 if train_buf else 0)} "
        f"val_shards={val_written + (1 if val_buf_tokens else 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
