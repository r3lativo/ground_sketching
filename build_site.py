#!/usr/bin/env python3
"""Export a curated subset of output/ conversations into docs/ as a static site.

Run inside the uv venv (needs Pillow):
    source .venv-site/bin/activate && python3 build_site.py
"""
import csv
import glob
import json
import os
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"
MAX_WIDTH = 900

# (category, conversation_id) pairs to publish. Edit this list to swap in a
# different subset, then rerun the script.
CONVERSATIONS = [
    ("Inferred_VA", "262_163_290_15"),
    ("Attributive_VA", "270_327_290_225"),
    ("Temporal_VA", "52_244_210_229"),
    ("Spatial_VA", "238_372_80_290"),
]

# The CSVs reference category folders with this casing; the folders on disk
# were originally downloaded lowercase. Fix is a rename, not a path rewrite.
CATEGORY_RENAMES = {
    "inferred_va": "Inferred_VA",
    "attributive_va": "Attributive_VA",
    "temporal_va": "Temporal_VA",
    "spatial_va": "Spatial_VA",
}


def fix_category_casing():
    entries = os.listdir(OUTPUT_DIR)
    for old, new in CATEGORY_RENAMES.items():
        if new in entries:
            continue
        if old in entries:
            tmp = OUTPUT_DIR / f"{old}__renaming_tmp"
            os.rename(OUTPUT_DIR / old, tmp)
            os.rename(tmp, OUTPUT_DIR / new)
            print(f"renamed output/{old} -> output/{new}")


def extract_seq_num(path):
    m = re.match(r".*_seq(\d+)\..+", os.path.basename(path))
    return int(m.group(1)) if m else 0


def resolve_sequence(img_path_rel, final_prompt_text):
    """Port of get_image_sequence() from data_visualization.py, operating on
    plain repo-relative path strings instead of a DataFrame row."""
    full_path = ROOT / img_path_rel
    if not full_path.exists():
        return []

    directory = full_path.parent
    filename = full_path.name
    match = re.match(r"(.*)_seq(\d+)(\..+)$", filename)

    if not match:
        return [{
            "path": str(full_path.relative_to(ROOT)),
            "prompt": final_prompt_text.strip() if final_prompt_text else "",
            "seq_id": "Final",
            "is_current": True,
        }]

    prefix, current_seq_num, ext = match.group(1), int(match.group(2)), match.group(3)
    pattern = str(directory / f"{prefix}_seq*{ext}")
    found_files = sorted(glob.glob(pattern), key=extract_seq_num)
    prompt_parts = final_prompt_text.split("$$$") if final_prompt_text else []

    sequence = []
    for i, file_p in enumerate(found_files):
        s_num = extract_seq_num(file_p)
        p_text = prompt_parts[i].strip() if i < len(prompt_parts) else ""
        sequence.append({
            "path": str(Path(file_p).relative_to(ROOT)),
            "prompt": p_text,
            "seq_id": f"Seq {s_num}",
            "is_current": s_num == current_seq_num,
        })
    return sequence


def compress_image(src: Path, dest: Path):
    # These are diffusion-model renders with visible grain/texture, not flat
    # vector art, so JPEG beats PNG by ~3x here with no visible quality loss
    # at demo resolution (verified by side-by-side comparison).
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > MAX_WIDTH:
            ratio = MAX_WIDTH / im.width
            im = im.resize((MAX_WIDTH, round(im.height * ratio)), Image.LANCZOS)
        im.save(dest, format="JPEG", quality=80, optimize=True)


def export_conversation(category, conv_id, image_registry, size_totals):
    csv_path = OUTPUT_DIR / category / conv_id / f"{conv_id}_aug.csv"

    def register_image(src_repo_rel):
        if src_repo_rel in image_registry:
            return image_registry[src_repo_rel]
        src_full = ROOT / src_repo_rel
        parts = Path(src_repo_rel).parts  # output / <Category> / <id> / images / <char> / file
        char, filename = parts[-2], parts[-1]
        filename = Path(filename).with_suffix(".jpg").name
        dest_rel = f"images/{conv_id}/{char}/{filename}"
        dest_full = DOCS_DIR / dest_rel
        compress_image(src_full, dest_full)
        size_totals["before"] += src_full.stat().st_size
        size_totals["after"] += dest_full.stat().st_size
        image_registry[src_repo_rel] = dest_rel
        return dest_rel

    rows_out = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            img_path = (r.get("img_path") or "").strip()
            final_prompt = r.get("final_prompt") or ""

            sequence = []
            out_img = None
            if img_path:
                raw_sequence = resolve_sequence(img_path, final_prompt)
                for item in raw_sequence:
                    dest_rel = register_image(item["path"])
                    sequence.append({**item, "path": dest_rel})
                    if item["is_current"]:
                        out_img = dest_rel
                if out_img is None and sequence:
                    out_img = sequence[-1]["path"]

            rows_out.append({
                "index": int(float(r.get("index") or 0)),
                "character": (r.get("character") or "").strip(),
                "text": r.get("text") or "",
                "mtype": (r.get("m-type") or "").strip(),
                "frame_choice": (r.get("frame_choice") or "").strip(),
                "frame_meta": (r.get("frame_meta") or "").strip(),
                "relation": (r.get("relation") or "").strip(),
                "extracted_triplets": (r.get("extracted_triplets") or "").strip(),
                "frame_id": (r.get("frame_id") or "").strip(),
                "imagery": (r.get("imagery") or "").strip(),
                "initial_prompt": (r.get("initial_prompt") or "").strip(),
                "final_prompt": (r.get("final_prompt") or "").strip(),
                "img_path": out_img,
                "sequence": sequence,
            })

    rows_out.sort(key=lambda row: row["index"])
    return rows_out


def main():
    fix_category_casing()

    (DOCS_DIR / "data").mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "images").mkdir(parents=True, exist_ok=True)

    image_registry = {}
    size_totals = {"before": 0, "after": 0}
    index_entries = []

    for category, conv_id in CONVERSATIONS:
        rows = export_conversation(category, conv_id, image_registry, size_totals)
        (DOCS_DIR / "data" / f"{conv_id}.json").write_text(
            json.dumps({"id": conv_id, "category": category, "rows": rows}, indent=None)
        )
        num_messages = sum(1 for row in rows if row["mtype"] == "text")
        index_entries.append({
            "id": conv_id,
            "category": category,
            "label": f"{category.replace('_VA', '')} / {conv_id}",
            "num_messages": num_messages,
        })
        print(f"exported {category}/{conv_id}: {len(rows)} rows, {num_messages} messages")

    (DOCS_DIR / "data" / "index.json").write_text(json.dumps(index_entries, indent=2))

    before_mb = size_totals["before"] / 1_000_000
    after_mb = size_totals["after"] / 1_000_000
    print(f"\n{len(image_registry)} unique images compressed")
    print(f"size before: {before_mb:.1f} MB -> after: {after_mb:.1f} MB "
          f"({100 * (1 - after_mb / before_mb):.0f}% smaller)" if before_mb else "no images")


if __name__ == "__main__":
    main()
