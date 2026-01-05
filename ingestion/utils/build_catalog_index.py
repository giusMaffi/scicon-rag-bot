#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


def read_catalog_products(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            # skip empty/broken lines
            if not row or not (row.get("product_handle") or "").strip():
                continue
            rows.append(row)
    return rows


def split_pipe(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x for x in s.split("|") if x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="catalog_products_dedup.csv")
    ap.add_argument("--output", required=True, help="catalog_index.json")
    args = ap.parse_args()

    rows = read_catalog_products(args.input)

    products_by_handle: dict[str, dict] = {}
    families: dict[str, list[str]] = defaultdict(list)
    editions: dict[str, list[str]] = defaultdict(list)
    categories: dict[str, list[str]] = defaultdict(list)
    support_items: list[str] = []

    for r in rows:
        handle = (r.get("product_handle") or "").strip()
        if not handle:
            continue

        type_guess = (r.get("type_guess") or "").strip()
        url = (r.get("pdp_url_canonical") or "").strip()

        fam_eye = (r.get("family_key_if_eyewear") or "").strip()
        fam_gen = (r.get("family_key_generic") or "").strip()
        var_gen = (r.get("variant_key_generic") or "").strip()

        # pick best family
        family_key = fam_eye or fam_gen or ""

        tag_category = split_pipe(r.get("collection_tags_category", ""))
        tag_edition = split_pipe(r.get("collection_tags_edition", ""))
        tag_support = split_pipe(r.get("collection_tags_support", ""))
        tag_model_family = split_pipe(r.get("collection_tags_model_family", ""))
        tag_merch = split_pipe(r.get("collection_tags_merchandising", ""))
        tag_all = split_pipe(r.get("collections_seen", ""))

        products_by_handle[handle] = {
            "handle": handle,
            "type": type_guess,
            "url": url,
            "family_key": family_key,
            "variant_key": var_gen,
            "tags": {
                "category": tag_category,
                "edition": tag_edition,
                "support": tag_support,
                "model_family": tag_model_family,
                "merchandising": tag_merch,
                "all_collections": tag_all
            }
        }

        if family_key:
            families[family_key].append(handle)

        for t in tag_edition:
            editions[t].append(handle)

        for t in tag_category:
            categories[t].append(handle)

        if tag_support:
            support_items.append(handle)

    # stable sort + uniq
    def uniq_sorted(lst: list[str]) -> list[str]:
        return sorted(set(lst))

    families = {k: uniq_sorted(v) for k, v in families.items()}
    editions = {k: uniq_sorted(v) for k, v in editions.items()}
    categories = {k: uniq_sorted(v) for k, v in categories.items()}
    support_items = uniq_sorted(support_items)

    out = {
        "meta": {
            "source": os.path.basename(args.input),
            "products_count": len(products_by_handle),
            "families_count": len(families),
            "editions_count": len(editions),
            "categories_count": len(categories),
            "support_items_count": len(support_items),
        },
        "products_by_handle": products_by_handle,
        "families": families,
        "editions": editions,
        "categories": categories,
        "support_items": support_items
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {args.output}")
    print(json.dumps(out["meta"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
