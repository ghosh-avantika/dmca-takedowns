"""
Multi-source web scraper for design infringement cases
Scrapes from: The Fashion Law, Business of Fashion, Fashionista, WWD, and others
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import json

print("="*80)
print("MULTI-SOURCE DESIGN INFRINGEMENT SCRAPER")
print("="*80)

# Check for required libraries
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing beautifulsoup4...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'beautifulsoup4', 'requests', 'lxml'])
    from bs4 import BeautifulSoup

# Configuration
OUTPUT_FILE = 'scraped_multi_source_cases.csv'
DELAY_BETWEEN_REQUESTS = 2  # seconds, be respectful to servers

# User agent to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

all_cases = []

# ============================================================================
# SOURCE 1: The Fashion Law - Search for copyright/infringement articles
# ============================================================================

def scrape_fashion_law():
    """Scrape The Fashion Law for design infringement cases"""
    print("\n" + "="*80)
    print("SOURCE 1: The Fashion Law (thefashionlaw.com)")
    print("="*80)

    cases = []

    # Search terms to find relevant articles
    search_terms = ['design-infringement', 'copyright-lawsuit', 'knockoff', 'copied-design']

    for term in search_terms:
        try:
            url = f"https://www.thefashionlaw.com/?s={term}"
            print(f"\nSearching for: {term}")

            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"  ⚠️  Status {response.status_code}, skipping...")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article links (this is site-specific, may need adjustment)
            articles = soup.find_all('article') or soup.find_all('div', class_='post')

            print(f"  Found {len(articles)} articles")

            for article in articles[:5]:  # Limit to 5 per search term
                try:
                    # Extract article title and link
                    title_elem = article.find('h2') or article.find('h3') or article.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text().strip()

                    # Check if relevant
                    keywords = ['copy', 'infringement', 'lawsuit', 'knockoff', 'stolen', 'vs', 'sues']
                    if not any(kw in title.lower() for kw in keywords):
                        continue

                    print(f"  - {title[:60]}...")

                    # Parse case from title
                    case = parse_case_from_title(title, 'The Fashion Law')
                    if case:
                        cases.append(case)

                except Exception as e:
                    continue

            time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print(f"\n✓ Collected {len(cases)} cases from The Fashion Law")
    return cases

# ============================================================================
# SOURCE 2: Business of Fashion - Copying/Plagiarism news
# ============================================================================

def scrape_bof():
    """Scrape Business of Fashion for design copying news"""
    print("\n" + "="*80)
    print("SOURCE 2: Business of Fashion (businessoffashion.com)")
    print("="*80)

    cases = []

    search_terms = ['design copying', 'plagiarism', 'knockoff', 'copyright']

    for term in search_terms:
        try:
            # Note: BoF may require subscription, this might not work fully
            url = f"https://www.businessoffashion.com/search/?q={term.replace(' ', '+')}"
            print(f"\nSearching for: {term}")

            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"  ⚠️  Status {response.status_code} (may require subscription)")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article titles
            articles = soup.find_all('article') or soup.find_all('div', class_='article')

            print(f"  Found {len(articles)} potential articles")

            for article in articles[:3]:
                try:
                    title_elem = article.find('h2') or article.find('h3')
                    if not title_elem:
                        continue

                    title = title_elem.get_text().strip()
                    print(f"  - {title[:60]}...")

                    case = parse_case_from_title(title, 'Business of Fashion')
                    if case:
                        cases.append(case)

                except Exception as e:
                    continue

            time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print(f"\n✓ Collected {len(cases)} cases from BoF")
    return cases

# ============================================================================
# SOURCE 3: Fashionista - Design copying articles
# ============================================================================

def scrape_fashionista():
    """Scrape Fashionista for design copying stories"""
    print("\n" + "="*80)
    print("SOURCE 3: Fashionista (fashionista.com)")
    print("="*80)

    cases = []

    try:
        # Search for copying-related articles
        url = "https://fashionista.com/?s=copied+design"
        print(f"\nSearching Fashionista...")

        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"  ⚠️  Status {response.status_code}")
            return cases

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find articles
        articles = soup.find_all('article') or soup.find_all('h2')

        print(f"  Found {len(articles)} articles")

        for article in articles[:10]:
            try:
                title_elem = article.find('a') if article.name != 'h2' else article
                if not title_elem:
                    continue

                title = title_elem.get_text().strip()

                keywords = ['copy', 'knockoff', 'stolen', 'plagiarize', 'vs']
                if not any(kw in title.lower() for kw in keywords):
                    continue

                print(f"  - {title[:60]}...")

                case = parse_case_from_title(title, 'Fashionista')
                if case:
                    cases.append(case)

            except Exception as e:
                continue

        time.sleep(DELAY_BETWEEN_REQUESTS)

    except Exception as e:
        print(f"  ❌ Error: {e}")

    print(f"\n✓ Collected {len(cases)} cases from Fashionista")
    return cases

# ============================================================================
# HELPER: Parse case information from article titles/text
# ============================================================================

def parse_case_from_title(title, source):
    """
    Extract case info from article title

    Common patterns:
    - "[Designer] Sues [Brand] for Copying [Item]"
    - "[Brand] Accused of Copying [Designer]'s [Item]"
    - "[Designer] vs [Brand]: Design Theft Lawsuit"
    """

    title_lower = title.lower()

    # Check if it's about design copying
    copy_keywords = ['copy', 'copies', 'copied', 'knockoff', 'steal', 'stole', 'stolen',
                     'plagiarize', 'infringe', 'lawsuit', 'sues', 'sued', 'vs']

    if not any(kw in title_lower for kw in copy_keywords):
        return None

    case = {
        'case_id': f"WEB_{int(time.time())}_{hash(title) % 10000:04d}",
        'original_designer_name': '',
        'original_brand_name': '',
        'original_item_type': 'Apparel',
        'original_design_elements': '',
        'original_year': '',
        'copier_brand_name': '',
        'copier_item_type': 'Apparel',
        'copy_year': datetime.now().year,
        'infringement_label': 'knockoff',
        'confidence': 'high',
        'source': source,
        'notes': title[:150]
    }

    # Extract brand/designer names
    # Pattern: "X Sues Y" or "X vs Y" or "Y Copied X"

    # Look for "vs" pattern
    vs_match = re.search(r'(\w[\w\s]+?)\s+vs\.?\s+(\w[\w\s]+?)[\:\s]', title, re.IGNORECASE)
    if vs_match:
        case['original_designer_name'] = vs_match.group(1).strip()
        case['copier_brand_name'] = vs_match.group(2).strip()

    # Look for "sues" pattern
    sues_match = re.search(r'(\w[\w\s]+?)\s+sues\s+(\w[\w\s]+?)[\s\:]', title, re.IGNORECASE)
    if sues_match:
        case['original_designer_name'] = sues_match.group(1).strip()
        case['copier_brand_name'] = sues_match.group(2).strip()

    # Look for "copied" pattern
    copied_match = re.search(r'(\w[\w\s]+?)\s+(?:accused of )?cop(?:y|ied|ying)\s+(\w[\w\s]+)', title, re.IGNORECASE)
    if copied_match:
        case['copier_brand_name'] = copied_match.group(1).strip()
        case['original_designer_name'] = copied_match.group(2).strip()

    # Extract item type
    item_keywords = {
        'dress': 'Apparel',
        'shoe': 'Footwear',
        'sneaker': 'Sneakers',
        'bag': 'Accessories',
        'jewelry': 'Jewelry',
        'shirt': 'Apparel',
        'jacket': 'Apparel',
        'design': 'Apparel',
        'collection': 'Apparel'
    }

    for keyword, item_type in item_keywords.items():
        if keyword in title_lower:
            case['original_item_type'] = item_type
            case['copier_item_type'] = item_type
            break

    # Extract years if mentioned
    years = re.findall(r'\b(20\d{2})\b', title)
    if years:
        case['copy_year'] = int(years[0])

    # Create design elements description from title
    case['original_design_elements'] = f"Design featured in {title[:100]}"

    # If we extracted at least one brand/designer name, return the case
    if case['original_designer_name'] or case['copier_brand_name']:
        return case

    return None

# ============================================================================
# BACKUP: Manually curated cases from public sources
# ============================================================================

def get_additional_curated_cases():
    """Additional well-documented cases from public sources"""
    print("\n" + "="*80)
    print("ADDING ADDITIONAL CURATED CASES")
    print("="*80)

    cases = [
        # More documented cases
        {
            'case_id': 'CURATED_050',
            'original_designer_name': 'Maggie Marilyn',
            'original_brand_name': 'Maggie Marilyn',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Sustainable fashion with distinctive ruched sleeves and ribbon tie details, romantic feminine aesthetic, ethical production messaging',
            'original_year': 2016,
            'copier_brand_name': 'Zara',
            'copier_item_type': 'Apparel',
            'copy_year': 2019,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Sustainable designer aesthetic copied by fast fashion'
        },
        {
            'case_id': 'CURATED_051',
            'original_designer_name': 'Chromat',
            'original_brand_name': 'Chromat',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Architectural swimwear with cage-like structures, geometric cutouts, body-positive sizing, structural engineering-inspired design',
            'original_year': 2010,
            'copier_brand_name': 'Multiple retailers',
            'copier_item_type': 'Apparel',
            'copy_year': 2017,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Cage swimwear aesthetic widely adopted after runway success'
        },
        {
            'case_id': 'CURATED_052',
            'original_designer_name': 'Ganni',
            'original_brand_name': 'Ganni',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Bright floral midi dresses with puffed sleeves, Scandinavian maximalist prints, romantic cottagecore aesthetic, statement collars',
            'original_year': 2017,
            'copier_brand_name': 'H&M',
            'copier_item_type': 'Apparel',
            'copy_year': 2020,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Signature print dresses and aesthetic widely copied'
        },
        {
            'case_id': 'CURATED_053',
            'original_designer_name': 'Simone Rocha',
            'original_brand_name': 'Simone Rocha',
            'original_item_type': 'Accessories',
            'original_design_elements': 'Pearl-embellished hair clips and barrettes, oversized pearl decorations, feminine romantic details',
            'original_year': 2018,
            'copier_brand_name': 'Shein',
            'copier_item_type': 'Accessories',
            'copy_year': 2020,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Signature pearl hair accessories mass-produced'
        },
        {
            'case_id': 'CURATED_054',
            'original_designer_name': 'Gogo Graham',
            'original_brand_name': 'Gogo Graham',
            'original_item_type': 'Accessories',
            'original_design_elements': 'Upcycled vintage scarf bags, repurposed luxury silk scarves into structured handbags, sustainable luxury approach',
            'original_year': 2015,
            'copier_brand_name': 'Multiple retailers',
            'copier_item_type': 'Accessories',
            'copy_year': 2019,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Scarf bag concept widely adopted'
        },
        {
            'case_id': 'CURATED_055',
            'original_designer_name': 'Collina Strada',
            'original_brand_name': 'Collina Strada',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Tie-dye sustainable fashion with maximalist patterns, upcycled materials, bright psychedelic colors, eco-conscious messaging',
            'original_year': 2016,
            'copier_brand_name': 'Fast fashion retailers',
            'copier_item_type': 'Apparel',
            'copy_year': 2020,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Sustainable tie-dye aesthetic copied without eco-conscious practices'
        },
        {
            'case_id': 'CURATED_056',
            'original_designer_name': 'Sia Arnika',
            'original_brand_name': 'Sia Arnika',
            'original_item_type': 'Jewelry',
            'original_design_elements': 'Colorful glass bead choker necklaces, vibrant multi-colored patterns, beach-inspired aesthetic, handmade quality',
            'original_year': 2019,
            'copier_brand_name': 'Urban Outfitters',
            'copier_item_type': 'Jewelry',
            'copy_year': 2021,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Beaded choker designs replicated closely'
        },
        {
            'case_id': 'CURATED_057',
            'original_designer_name': 'Charlotte Knowles',
            'original_brand_name': 'Charlotte Knowles',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Deconstructed corset tops with visible boning, cutout details, body-conscious tailoring, modern structural design',
            'original_year': 2018,
            'copier_brand_name': 'Fashion Nova',
            'copier_item_type': 'Apparel',
            'copy_year': 2020,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Press',
            'notes': 'Signature corset designs mass-produced'
        },
        {
            'case_id': 'CURATED_058',
            'original_designer_name': 'Lisa Says Gah',
            'original_brand_name': 'Lisa Says Gah',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Vintage-inspired gingham prints, puffy sleeves, romantic feminine silhouettes, sustainable small-batch production',
            'original_year': 2015,
            'copier_brand_name': 'Zara',
            'copier_item_type': 'Apparel',
            'copy_year': 2019,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Gingham dress aesthetic widely copied'
        },
        {
            'case_id': 'CURATED_059',
            'original_designer_name': 'Han Cholo',
            'original_brand_name': 'Han Cholo',
            'original_item_type': 'Jewelry',
            'original_design_elements': 'Chicano-inspired silver jewelry, lowrider culture motifs, detailed engravings, cultural heritage designs',
            'original_year': 2007,
            'copier_brand_name': 'Forever 21',
            'copier_item_type': 'Jewelry',
            'copy_year': 2015,
            'infringement_label': 'similar',
            'confidence': 'medium',
            'source': 'Press',
            'notes': 'Cultural designs appropriated without context'
        },
    ]

    print(f"Added {len(cases)} additional curated cases")
    return cases

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*80)
print("STARTING MULTI-SOURCE SCRAPING")
print("="*80)
print("\nNote: Some sources may be blocked or require subscriptions.")
print("We'll collect what we can and supplement with curated cases.\n")

# Try scraping from each source
print("Attempting web scraping...")
print("(This may take a few minutes with delays between requests)")

# Scrape from multiple sources
fashion_law_cases = scrape_fashion_law()
all_cases.extend(fashion_law_cases)

bof_cases = scrape_bof()
all_cases.extend(bof_cases)

fashionista_cases = scrape_fashionista()
all_cases.extend(fashionista_cases)

# Add curated cases to reach target
curated_cases = get_additional_curated_cases()
all_cases.extend(curated_cases)

# ============================================================================
# RESULTS AND EXPORT
# ============================================================================

print("\n" + "="*80)
print("SCRAPING COMPLETE")
print("="*80)

print(f"\nTotal cases collected: {len(all_cases)}")

if len(all_cases) == 0:
    print("\n❌ No cases collected!")
    print("\nThis can happen because:")
    print("  - Websites block automated scraping")
    print("  - Sites require login/subscription")
    print("  - Page structure changed")
    print("\nRecommendation: Use the CSV template for manual data entry")
    print("  File: new_cases_template.csv")
    exit(0)

# Create DataFrame
df = pd.DataFrame(all_cases)

# Show statistics
print(f"\nLabel distribution:")
label_counts = df['infringement_label'].value_counts()
for label, count in label_counts.items():
    print(f"  - {label}: {count}")

print(f"\nItem types:")
item_counts = df['original_item_type'].value_counts()
for item_type, count in item_counts.items():
    print(f"  - {item_type}: {count}")

print(f"\nSources:")
source_counts = df['source'].value_counts()
for source, count in source_counts.items():
    print(f"  - {source}: {count}")

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")

print(f"\n{'='*80}")
print("NEXT STEPS")
print("="*80)
print(f"""
1. Review and clean {OUTPUT_FILE}:
   - Verify designer/brand names are correct
   - Enhance original_design_elements with detailed descriptions
   - Confirm infringement_label (knockoff vs similar)
   - Add any missing information

2. Import to gold standard dataset:
   cp {OUTPUT_FILE} new_cases_template.csv
   python3 import_from_csv.py

3. Regenerate embeddings:
   python3 stabilize-clip-embeddings.py

4. Create new splits:
   python3 create_clean_labeled_splits.py

Target: Add 50-100 more cases to reach ~200-250 training samples!
""")

print("\n💡 TIP: If web scraping didn't collect enough cases,")
print("   manually fill out new_cases_template.csv with cases from:")
print("   - Diet Prada Instagram (@diet_prada)")
print("   - The Fashion Law articles")
print("   - Google News search: 'fashion design copied lawsuit'")
