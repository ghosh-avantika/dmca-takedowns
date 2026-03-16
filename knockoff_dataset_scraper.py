# knockoff_dataset_scraper.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import spacy
from tqdm import tqdm

# ----------------------------
# 1. Load NLP model for entity extraction
# ----------------------------
# Use spaCy small English model
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# 2. List of low-risk URLs to scrape
# ----------------------------
# Add more URLs to this list as you discover them
urls_to_scrape = [
    "https://designmilk.com/indie-designer-copied-example",
    "https://core77.com/blog/design-knockoff-example",
    "https://www.reddit.com/r/Etsy/comments/example_thread"
]

# ----------------------------
# 3. Fetch HTML content
# ----------------------------
def fetch_page(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; knockoff-scraper/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# ----------------------------
# 4. Extract text excerpt from page
# ----------------------------
def extract_excerpt(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")
    # Limit to first 3 paragraphs (~200 words)
    text = " ".join([p.get_text().strip() for p in paragraphs[:3]])
    return text

# ----------------------------
# 5. Parse the text into dataset fields
# ----------------------------
def parse_fields(text, url):
    doc = nlp(text)

    # Helper functions
    def extract_names(doc):
        names = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
        return names if names else None

    def extract_years(text):
        years = re.findall(r"\b(19|20)\d{2}\b", text)
        return years if years else None

    def extract_item_type(text):
        product_nouns = ["bag", "shirt", "table", "chair", "dress", "jewelry", "lamp"]
        matches = [word for word in product_nouns if word in text.lower()]
        return matches[0] if matches else None

    names = extract_names(doc)

    original_designer_name = names[0] if names else None
    original_brand_name = names[1] if len(names) > 1 else None
    original_item_type = extract_item_type(text)

    # Detect copier
    copier_brand_name = None
    copier_match = re.search(r"(?:copied|replicated|knockoff|stolen) by ([A-Z][a-zA-Z0-9\s]+)", text)
    if copier_match:
        copier_brand_name = copier_match.group(1)
    copier_item_type = original_item_type

    # Extract years
    years = extract_years(text)
    original_year = years[0] if years else None
    copy_year = years[1] if len(years) > 1 else None

    # Design elements: sentences mentioning "design" or "pattern"
    design_elements = ""
    sentences = re.split(r'\.|\n', text)
    for s in sentences:
        if "design" in s.lower() or "pattern" in s.lower():
            design_elements = s.strip()
            break

    # Infringement label
    infringement_label = 1 if any(k in text.lower() for k in ["copied", "knockoff", "replica", "stolen"]) else 0

    # Confidence scoring
    if infringement_label:
        if "takedown" in text.lower() or "lawsuit" in text.lower():
            confidence = 0.9
        elif "side-by-side" in text.lower() or "photo" in text.lower():
            confidence = 0.7
        else:
            confidence = 0.5
    else:
        confidence = 0.2

    # Notes: first 200 characters
    notes = text[:200]

    return {
        "original_designer_name": original_designer_name,
        "original_brand_name": original_brand_name,
        "original_item_type": original_item_type,
        "original_design_elements": design_elements,
        "original_year": original_year,
        "copier_brand_name": copier_brand_name,
        "copier_item_type": copier_item_type,
        "copy_year": copy_year,
        "infringement_label": infringement_label,
        "confidence": confidence,
        "source": url,
        "notes": notes
    }

# ----------------------------
# 6. Scrape all URLs and build dataset
# ----------------------------
dataset = []
for url in tqdm(urls_to_scrape):
    html = fetch_page(url)
    if html:
        text = extract_excerpt(html)
        row = parse_fields(text, url)
        dataset.append(row)
    time.sleep(5)  # respectful delay

# ----------------------------
# 7. Export to CSV
# ----------------------------
df = pd.DataFrame(dataset)
df.to_csv("knockoff_dataset.csv", index=False)
print("✅ Scraping complete! Dataset saved as knockoff_dataset.csv")
