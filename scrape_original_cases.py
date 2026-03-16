"""
Scrape "original" design cases from Reddit and append to negative_cases_originals.csv.

Heuristics: require phrases like "my design" or "original" in title/body.
"""

import csv
import time
import re
from datetime import datetime
from pathlib import Path

import requests


OUTPUT_FILE = "negative_cases_originals.csv"
DELAY_BETWEEN_REQUESTS = 2
MAX_TOTAL_RESULTS = 200

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

ORIGINAL_KEYWORDS = ["my design", "original", "origin"]
COPY_KEYWORDS = [
    "copy", "copies", "copied", "knockoff", "steal", "stolen", "plagiar",
    "infringe", "lawsuit", "sues", "sued", "vs", "rip off", "ripoff",
    "replica", "identical"
]

SUBREDDITS = [
    "fashion", "streetwear", "malefashionadvice", "femalefashionadvice",
    "Designer", "fashiondesign", "sewing", "streetwearstartup"
]


def is_original_text(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in COPY_KEYWORDS):
        return False
    return any(k in t for k in ORIGINAL_KEYWORDS)


def parse_original_from_text(title: str, body: str, source: str) -> dict:
    case = {
        "case_id": f"ORIG_WEB_{int(time.time())}_{abs(hash(title)) % 10000:04d}",
        "original_designer_name": "",
        "original_brand_name": "",
        "original_item_type": "Apparel",
        "original_design_elements": "",
        "original_year": "",
        "copier_brand_name": "N/A - Original Design",
        "copier_item_type": "",
        "copy_year": "",
        "infringement_label": "original",
        "confidence": "high",
        "source": source,
        "notes": title[:200],
    }

    # crude designer/brand extraction: first two capitalized words
    words = re.findall(r"[A-Z][a-zA-Z]+", title)
    if words:
        designer = " ".join(words[:2])
        case["original_designer_name"] = designer
        case["original_brand_name"] = designer

    # derive a design element summary
    summary = title
    if body:
        summary = f"{title}. {body[:140]}"
    case["original_design_elements"] = f"Original design described in: {summary[:180]}"

    years = re.findall(r"\b(20\d{2})\b", title)
    if years:
        case["original_year"] = int(years[0])

    return case


def scrape_reddit():
    cases = []
    query = '"my design" OR original OR origin'
    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/search.json?q={query}&restrict_sr=1&sort=new&limit=100"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "").strip()
            body = p.get("selftext", "").strip()
            combined = f"{title} {body}"
            if not combined or not is_original_text(combined):
                continue
            cases.append(parse_original_from_text(title, body, f"Reddit:r/{sub}"))
            if len(cases) >= MAX_TOTAL_RESULTS:
                return cases
        time.sleep(DELAY_BETWEEN_REQUESTS)
    return cases


def append_to_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print("No new original cases found.")
        return

    fieldnames = [
        "case_id", "original_designer_name", "original_brand_name", "original_item_type",
        "original_design_elements", "original_year", "copier_brand_name", "copier_item_type",
        "copy_year", "infringement_label", "confidence", "source", "notes"
    ]

    existing_ids = set()
    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row.get("case_id"))

    new_rows = [r for r in rows if r["case_id"] not in existing_ids]
    if not new_rows:
        print("No new unique rows to append.")
        return

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        for r in new_rows:
            writer.writerow(r)

    print(f"Appended {len(new_rows)} rows to {path.name}")


def main():
    print("=" * 80)
    print("SCRAPING ORIGINAL CASES")
    print("=" * 80)

    all_cases = scrape_reddit()

    append_to_csv(Path(OUTPUT_FILE), all_cases)


if __name__ == "__main__":
    main()
