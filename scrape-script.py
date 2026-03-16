import requests
import time
import csv
import uuid
import re

TARGET_CASES = 150
REQUEST_DELAY = 2.0
OUTPUT_FILE = "scraped_cases_text_only.csv"

SEARCH_QUERIES = [
    "copied design",
    "fashion knockoff",
    "brand copied",
    "design rip off",
    "inspired by brand"
]

SUBREDDITS = ["fashion", "streetwear", "graphic_design"]

HEADERS = {
    "User-Agent": "MLP-Design-Similarity/1.1 (academic research)"
}

def is_valid_description(text):
    if not text:
        return False
    words = text.split()
    if len(words) < 40:
        return False
    if text.lower().count("http") > 2:
        return False
    return True

def fetch_top_comment(permalink):
    try:
        url = f"https://www.reddit.com{permalink}.json"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        comments = r.json()[1]["data"]["children"]
        if comments:
            return comments[0]["data"].get("body", "")
    except Exception:
        pass
    return ""

def scrape_cases():
    cases = []
    seen = set()

    for query in SEARCH_QUERIES:
        for subreddit in SUBREDDITS:
            if len(cases) >= TARGET_CASES:
                return cases

            search_url = (
                f"https://www.reddit.com/r/{subreddit}/search.json"
                f"?q={query}&restrict_sr=1&limit=50"
            )

            try:
                r = requests.get(search_url, headers=HEADERS)
                r.raise_for_status()
                posts = r.json()["data"]["children"]
            except Exception:
                continue

            for post in posts:
                if len(cases) >= TARGET_CASES:
                    break

                data = post["data"]
                pid = data.get("id")
                if pid in seen:
                    continue
                seen.add(pid)

                title = data.get("title", "")
                body = data.get("selftext", "")
                text = f"{title}. {body}".strip()

                if not is_valid_description(text):
                    comment = fetch_top_comment(data.get("permalink", ""))
                    text = f"{title}. {comment}".strip()

                if not is_valid_description(text):
                    continue  # hard drop

                cases.append({
                    "case_id": f"SCR_{uuid.uuid4().hex[:8]}",
                    "description": text,
                    "source": "reddit",
                    "query": query,
                    "url": f"https://reddit.com{data.get('permalink')}"
                })

                time.sleep(REQUEST_DELAY)

    return cases

def save_csv(cases):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "description", "source", "query", "url"]
        )
        writer.writeheader()
        for c in cases:
            writer.writerow(c)

if __name__ == "__main__":
    print("Scraping high-quality text-only cases...")
    cases = scrape_cases()
    save_csv(cases)

    print(f"✓ Saved {len(cases)} usable cases to {OUTPUT_FILE}")
