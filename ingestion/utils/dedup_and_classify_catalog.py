#!/usr/bin/env python3
"""
Dedup + classify + family grouping + typed collections (edition/category/merch/support/model_family).

Input: CSV with 1 column (no header) containing URLs.
Output:
 - ingestion/out/catalog_products_dedup.csv
 - ingestion/out/catalog_families_eyewear.csv
 - ingestion/out/report_summary.json
 - ingestion/out/report_summary.txt

Goals:
 - canonicalize URLs
 - deduplicate by product handle (Shopify /products/<handle>)
 - detect product type guess: eyewear/accessory/bag/blue_light/unknown
 - build eyewear family key (best-effort from collection segments)
 - IMPORTANT: collections are typed via external mapping to preserve "edition/capsule" (also for eyewear),
              and to prevent support/spare parts being misclassified as bags/eyewear.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

SCICON_DOMAIN = "sciconsports.com"

# --- heuristics keywords (IT/EN mixed) ---
KW_BAGS = [
    "bags", "bag", "zaini", "zaino", "borsa", "borse", "backpack", "travel", "luggage"
]
KW_ACCESSORIES = [
    "accessori", "accessory", "accessories", "spare", "replacement", "parti-di-ricambio",
    "ricambi", "nose", "nasello", "temple", "aste", "viti", "clip-in", "clipin", "lens", "lenti",
    "case", "custodia", "kit"
]
KW_BLUE_LIGHT = [
    "x-blue", "xblue", "blue", "computer-glasses", "glasses-blue", "anti-luce-blu"
]
KW_EYEWEAR = [
    "occhiali", "eyewear", "sunglasses", "sportivi-da-sole", "ciclismo",
    "occhiali-da-ciclismo", "occhiali-sportivi-da-sole", "occhiali-da-vista",
    "occhiali-da-vista-sport", "vista"
]

# Some collections explicitly indicate eyewear model families:
# e.g. /collections/aerostorm-sunglasses, /collections/aeroshade-kunken-sunglasses
RE_EYEWEAR_COLLECTION_MODEL = re.compile(r"^aero[a-z0-9-]+-(sunglasses|eyewear)$", re.IGNORECASE)
RE_PRODUCT_PATH = re.compile(r"^/collections/.*/products/([^/?#]+)$", re.IGNORECASE)
RE_PRODUCTS_GENERIC = re.compile(r"^/products/([^/?#]+)$", re.IGNORECASE)
RE_COLLECTIONS = re.compile(r"^/collections/([^/?#]+)$", re.IGNORECASE)

# Generic family extraction for non-eyewear handles (bags/apparel/accessory, etc.)
# Example: zaino-pro-35l-pr070000516 -> family: zaino-pro-35l ; variant: pr070000516
RE_GENERIC_VARIANT_SUFFIX = re.compile(r"^(?P<family>.+)-(?P<variant>[a-z]{1,4}\d{5,})$", re.IGNORECASE)


@dataclass
class ParsedURL:
    raw: str
    is_scicon: bool
    kind: str  # product | collection | page | other
    canonical: str
    product_handle: str | None
    collection_handle: str | None
    path: str
    query: str


def normalize_url(u: str) -> str:
    """
    Normalize:
      - strip spaces
      - add https if missing scheme
      - force sciconsports.com lowercase (netloc)
      - remove all query params
      - remove trailing slash (except root)
    """
    u = (u or "").strip()
    if not u:
        return u

    # Fix common "http://sciconsports.com" -> https
    if u.startswith("http://"):
        u = "https://" + u[len("http://"):]
    if not (u.startswith("https://") or u.startswith("http://")):
        u = "https://" + u.lstrip("/")

    parsed = urlparse(u)
    netloc = parsed.netloc.lower().replace("www.", "")

    # Remove all query params for now
    query = ""

    # Normalize path
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    canonical = urlunparse(("https", netloc, path, "", query, ""))
    return canonical


def classify_from_path(path: str) -> str:
    p = path.lower()

    def has_any(keywords: list[str]) -> bool:
        return any(k in p for k in keywords)

    # Priority: blue light pages
    if has_any(KW_BLUE_LIGHT):
        return "blue_light"

    # Bags
    if has_any(KW_BAGS):
        return "bag"

    # Accessories / spare parts
    if has_any(KW_ACCESSORIES):
        return "accessory"

    # Eyewear
    if has_any(KW_EYEWEAR):
        return "eyewear"

    return "unknown"


def parse_scicon_url(raw: str) -> ParsedURL:
    canonical = normalize_url(raw)
    if not canonical:
        return ParsedURL(
            raw=raw, is_scicon=False, kind="empty", canonical="",
            product_handle=None, collection_handle=None, path="", query=""
        )

    parsed = urlparse(canonical)
    is_scicon = (parsed.netloc.replace("www.", "") == SCICON_DOMAIN)

    path = parsed.path or ""
    query = parsed.query or ""

    product_handle = None
    collection_handle = None
    kind = "other"

    m = RE_PRODUCT_PATH.match(path)
    if m:
        kind = "product"
        product_handle = m.group(1)
        parts = [x for x in path.split("/") if x]
        # ['collections', '<collection>', 'products', '<product>']
        if len(parts) >= 4 and parts[0] == "collections":
            collection_handle = parts[1]
    else:
        m2 = RE_PRODUCTS_GENERIC.match(path)
        if m2:
            kind = "product"
            product_handle = m2.group(1)
        else:
            mc = RE_COLLECTIONS.match(path)
            if mc:
                kind = "collection"
                collection_handle = mc.group(1)
            elif path.startswith("/pages/"):
                kind = "page"
            else:
                kind = "other"

    return ParsedURL(
        raw=raw,
        is_scicon=is_scicon,
        kind=kind,
        canonical=canonical,
        product_handle=product_handle,
        collection_handle=collection_handle,
        path=path,
        query=query
    )


def eyewear_family_key(collection_handle: str | None, product_handle: str | None) -> str | None:
    """
    Best-effort family key for eyewear.
    If we have collection like "aerostorm-sunglasses" => "aerostorm"
    If collection like "aeroshade-kunken-sunglasses" => "aeroshade-kunken"
    Otherwise fallback to product_handle prefix before "-ey" (common in Scicon)
    """
    if collection_handle:
        ch = collection_handle.lower()
        if ch.endswith("-sunglasses"):
            return ch.replace("-sunglasses", "")
        if ch.endswith("-eyewear"):
            return ch.replace("-eyewear", "")
        if RE_EYEWEAR_COLLECTION_MODEL.match(ch):
            return re.sub(r"-(sunglasses|eyewear)$", "", ch)

    if product_handle:
        ph = product_handle.lower()
        if "-ey" in ph:
            return ph.split("-ey", 1)[0]
        return ph

    return None


def generic_family_key(product_handle: str) -> tuple[str | None, str | None]:
    """
    Generic family+variant extraction for non-eyewear.
    Returns (family, variant) or (None, None) if not detected.
    Example: zaino-pro-35l-pr070000516 -> ('zaino-pro-35l', 'pr070000516')
    """
    m = RE_GENERIC_VARIANT_SUFFIX.match(product_handle or "")
    if not m:
        return None, None
    family = m.group("family")
    variant = m.group("variant")
    if not family or not variant:
        return None, None
    return family, variant


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_urls_csv_onecol(path: str) -> list[str]:
    urls: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            u = (row[0] or "").strip()
            if u:
                urls.append(u)
    return urls


def load_collections_map(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def _startswith_any(s: str, prefixes: list[str]) -> bool:
    return any(s.startswith(p) for p in prefixes)


def _contains_any(s: str, needles: list[str]) -> bool:
    return any(n in s for n in needles)


def classify_collection_handle(collection_handle: str, cmap: dict) -> str:
    """
    Returns one of: merchandising | support | edition | model_family | category
    Default fallback: category
    """
    ch = (collection_handle or "").strip().lower()
    if not ch:
        return "category"

    merch_exact = set((cmap.get("merchandising_exact") or []))
    merch_prefixes = list((cmap.get("merchandising_prefixes") or []))

    support_prefixes = list((cmap.get("support_prefixes") or []))
    support_contains = list((cmap.get("support_contains") or []))

    edition_exact = set((cmap.get("edition_exact") or []))
    edition_contains = list((cmap.get("edition_contains") or []))

    mf_exact = set((cmap.get("model_family_exact") or []))
    mf_suffixes = list((cmap.get("model_family_suffixes") or []))

    # Merchandising
    if ch in merch_exact or _startswith_any(ch, merch_prefixes):
        return "merchandising"

    # Support / spare parts
    if _startswith_any(ch, support_prefixes) or _contains_any(ch, support_contains):
        return "support"

    # Edition / capsule
    if ch in edition_exact or _contains_any(ch, edition_contains):
        return "edition"

    # Model family (eyewear)
    if ch in mf_exact or any(ch.endswith(suf) for suf in mf_suffixes) or RE_EYEWEAR_COLLECTION_MODEL.match(ch):
        return "model_family"

    # Default: category
    return "category"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with one column, no header, URLs")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--collections-map",
        default="ingestion/config/collections_map.json",
        help="JSON map to type collection handles (edition/category/merch/support/model_family)"
    )
    args = ap.parse_args()

    in_path = args.input
    outdir = args.outdir
    ensure_outdir(outdir)

    cmap = load_collections_map(args.collections_map)

    raw_urls = read_urls_csv_onecol(in_path)
    parsed_items: list[ParsedURL] = [parse_scicon_url(u) for u in raw_urls]

    products = [p for p in parsed_items if p.kind == "product" and p.product_handle]

    # Dedup products by product_handle (canonical)
    dedup_by_handle: dict[str, dict] = {}

    for p in products:
        handle = p.product_handle  # type: ignore

        if handle not in dedup_by_handle:
            dedup_by_handle[handle] = {
                "product_handle": handle,
                "pdp_url_canonical": p.canonical,
                "paths_seen": set([p.path]),
                "collections_seen": set([p.collection_handle]) if p.collection_handle else set(),
                "raw_count": 1,
                "type_guess": classify_from_path(p.path),
            }
        else:
            dedup_by_handle[handle]["raw_count"] += 1
            dedup_by_handle[handle]["paths_seen"].add(p.path)
            if p.collection_handle:
                dedup_by_handle[handle]["collections_seen"].add(p.collection_handle)

            # if we had unknown but now path suggests something, upgrade
            cur = dedup_by_handle[handle]["type_guess"]
            new = classify_from_path(p.path)
            if cur == "unknown" and new != "unknown":
                dedup_by_handle[handle]["type_guess"] = new

    # Post-pass 1: type collections for each product + enforce support priority
    # We do it here so we have full collections_seen set.
    for handle, row in dedup_by_handle.items():
        collections_seen = [c for c in row["collections_seen"] if c]
        typed = {"category": set(), "edition": set(), "merchandising": set(), "support": set(), "model_family": set()}

        for c in collections_seen:
            t = classify_collection_handle(c, cmap)
            typed[t].add(c)

        row["collection_tags_category"] = typed["category"]
        row["collection_tags_edition"] = typed["edition"]
        row["collection_tags_merch"] = typed["merchandising"]
        row["collection_tags_support"] = typed["support"]
        row["collection_tags_model_family"] = typed["model_family"]

        # HARD RULE: support wins over everything for type_guess
        if typed["support"]:
            row["type_guess"] = "accessory"

    # Post-pass 2: if still unknown but handle looks eyewear by naming, set eyewear
    for handle, row in dedup_by_handle.items():
        if row["type_guess"] == "unknown" and handle.lower().startswith("aero"):
            row["type_guess"] = "eyewear"

    # Build eyewear families (unchanged logic, but now we prefer model_family collections, not editions/merch)
    families: dict[str, dict] = {}
    for handle, row in dedup_by_handle.items():
        if row["type_guess"] != "eyewear":
            continue

        # Prefer model-family tags for family extraction
        model_family_cols = sorted([c for c in row["collection_tags_model_family"] if c])
        other_cols = sorted([c for c in row["collections_seen"] if c])

        fam = None
        # First: model family collections
        for c in model_family_cols:
            fam = eyewear_family_key(c, handle)
            if fam:
                break
        # Fallback: any other collection (could be category; editions are OK but not ideal)
        if not fam:
            for c in other_cols:
                fam = eyewear_family_key(c, handle)
                if fam:
                    break
        # Final fallback
        if not fam:
            fam = eyewear_family_key(None, handle)

        if not fam:
            continue

        if fam not in families:
            families[fam] = {
                "family_key": fam,
                "products": [],
                "collections_seen": set(),
            }
        families[fam]["products"].append(handle)
        for c in other_cols:
            families[fam]["collections_seen"].add(c)

        row["family_key_if_eyewear"] = fam

    # Generic family for non-eyewear (bags/accessory/apparel/etc.)
    for handle, row in dedup_by_handle.items():
        if row["type_guess"] == "eyewear":
            row["family_key_generic"] = ""
            row["variant_key_generic"] = ""
            continue
        fam, var = generic_family_key(handle)
        row["family_key_generic"] = fam or ""
        row["variant_key_generic"] = var or ""

    # Write catalog_products_dedup.csv
    out_products_csv = os.path.join(outdir, "catalog_products_dedup.csv")
    with open(out_products_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "product_handle",
            "pdp_url_canonical",
            "type_guess",
            "family_key_if_eyewear",
            "family_key_generic",
            "variant_key_generic",
            "raw_count",
            "collection_tags_category",
            "collection_tags_edition",
            "collection_tags_merchandising",
            "collection_tags_support",
            "collection_tags_model_family",
            "collections_seen",
            "paths_seen",
        ])

        for handle in sorted(dedup_by_handle.keys()):
            row = dedup_by_handle[handle]
            w.writerow([
                handle,
                row["pdp_url_canonical"],
                row["type_guess"],
                row.get("family_key_if_eyewear", "") or "",
                row.get("family_key_generic", "") or "",
                row.get("variant_key_generic", "") or "",
                row["raw_count"],
                "|".join(sorted(row["collection_tags_category"])) if row.get("collection_tags_category") else "",
                "|".join(sorted(row["collection_tags_edition"])) if row.get("collection_tags_edition") else "",
                "|".join(sorted(row["collection_tags_merch"])) if row.get("collection_tags_merch") else "",
                "|".join(sorted(row["collection_tags_support"])) if row.get("collection_tags_support") else "",
                "|".join(sorted(row["collection_tags_model_family"])) if row.get("collection_tags_model_family") else "",
                "|".join(sorted(row["collections_seen"])) if row["collections_seen"] else "",
                "|".join(sorted(row["paths_seen"])) if row["paths_seen"] else "",
            ])

    # Write catalog_families_eyewear.csv
    out_families_csv = os.path.join(outdir, "catalog_families_eyewear.csv")
    with open(out_families_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family_key", "product_count", "products", "collections_seen"])
        for fam in sorted(families.keys()):
            data = families[fam]
            products_list = sorted(data["products"])
            w.writerow([
                fam,
                len(products_list),
                "|".join(products_list),
                "|".join(sorted(data["collections_seen"])) if data["collections_seen"] else "",
            ])

    # Summary report
    type_counts = {}
    for _, row in dedup_by_handle.items():
        type_counts[row["type_guess"]] = type_counts.get(row["type_guess"], 0) + 1

    top_duplicates = sorted(
        ((h, r["raw_count"]) for h, r in dedup_by_handle.items() if r["raw_count"] > 1),
        key=lambda x: x[1],
        reverse=True
    )[:30]

    summary = {
        "input_urls_total": len(raw_urls),
        "parsed_products_total": len(products),
        "dedup_products_unique": len(dedup_by_handle),
        "type_counts_on_dedup": type_counts,
        "eyewear_families_count": len(families),
        "top_duplicates_by_handle": [{"handle": h, "count": c} for h, c in top_duplicates],
        "collections_map_loaded": bool(cmap),
        "outputs": {
            "catalog_products_dedup": out_products_csv,
            "catalog_families_eyewear": out_families_csv
        }
    }

    out_json = os.path.join(outdir, "report_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(outdir, "report_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("SCICON CATALOG - DEDUP & CLASSIFY SUMMARY\n")
        f.write("========================================\n\n")
        f.write(f"Input URLs total: {summary['input_urls_total']}\n")
        f.write(f"Parsed products total: {summary['parsed_products_total']}\n")
        f.write(f"Unique products (dedup): {summary['dedup_products_unique']}\n\n")
        f.write("Type counts (dedup):\n")
        for k in sorted(type_counts.keys()):
            f.write(f"  - {k}: {type_counts[k]}\n")
        f.write(f"\nEyewear families: {summary['eyewear_families_count']}\n\n")
        f.write(f"Collections map loaded: {summary['collections_map_loaded']}\n\n")
        f.write("Top duplicate product handles (raw occurrences):\n")
        for item in summary["top_duplicates_by_handle"]:
            f.write(f"  - {item['handle']}: {item['count']}\n")

    print(f"[OK] Wrote: {out_products_csv}")
    print(f"[OK] Wrote: {out_families_csv}")
    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
