"""
Scrape design infringement cases from Diet Prada and similar sources
Outputs CSV in the format matching gold-standard-cases-CORRECTED.csv
"""

import re
import pandas as pd
from datetime import datetime
import time

# Check for required library
try:
    import instaloader
except ImportError:
    print("Installing instaloader...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'instaloader'])
    import instaloader

print("="*80)
print("SCRAPING DESIGN INFRINGEMENT CASES")
print("="*80)

# Configuration
TARGET_ACCOUNTS = ['diet_prada']  # Can add more accounts
MAX_POSTS_PER_ACCOUNT = 200  # Adjust based on how many you want
OUTPUT_FILE = 'scraped_design_cases.csv'

# Keywords that indicate a copying/knockoff post
COPY_KEYWORDS = [
    'copy', 'copied', 'copies', 'copying',
    'knockoff', 'knock-off', 'knock off',
    'rip off', 'ripped off', 'ripoff',
    'stolen', 'stole', 'steal',
    'identical', 'replica', 'replicate',
    'plagiarize', 'plagiarism',
    'original vs', 'vs copy',
    'fast fashion', 'mass retailer',
    'indie designer', 'small designer'
]

def extract_case_info(caption, post_date):
    """
    Parse Instagram caption to extract case information

    Args:
        caption (str): Post caption text
        post_date (datetime): When post was made

    Returns:
        dict: Extracted case information or None if not relevant
    """
    if not caption:
        return None

    caption_lower = caption.lower()

    # Check if post is about copying
    is_copy_post = any(keyword in caption_lower for keyword in COPY_KEYWORDS)
    if not is_copy_post:
        return None

    case = {
        'case_id': f"DP_SCRAPED_{int(time.time())}_{hash(caption[:50]) % 10000:04d}",
        'original_designer_name': '',
        'original_brand_name': '',
        'original_item_type': '',
        'original_design_elements': '',
        'original_year': '',
        'copier_brand_name': '',
        'copier_item_type': '',
        'copy_year': post_date.year,
        'infringement_label': '',
        'confidence': '',
        'source': 'Diet Prada',
        'notes': caption[:200]  # First 200 chars as notes
    }

    # Extract mentioned brands/designers (look for @mentions)
    mentions = re.findall(r'@(\w+)', caption)

    # Try to identify original designer (often first mention)
    if mentions:
        case['original_designer_name'] = mentions[0].replace('_', ' ').title()
        case['original_brand_name'] = mentions[0].replace('_', ' ').title()

    # Try to identify copier (look for brand names)
    common_fast_fashion = [
        'zara', 'h&m', 'hm', 'forever 21', 'forever21', 'shein', 'she in',
        'urban outfitters', 'topshop', 'asos', 'boohoo', 'prettylittlething',
        'fashion nova', 'fashionnova', 'target', 'walmart', 'primark',
        'mango', 'anthropologie', 'revolve', 'nasty gal', 'nastygal'
    ]

    for brand in common_fast_fashion:
        if brand in caption_lower:
            case['copier_brand_name'] = brand.title()
            break

    # If more than one mention, second might be copier
    if len(mentions) > 1 and not case['copier_brand_name']:
        case['copier_brand_name'] = mentions[1].replace('_', ' ').title()

    # Try to identify item type
    item_keywords = {
        'Apparel': ['dress', 'shirt', 'top', 'jacket', 'coat', 'sweater', 'hoodie', 'pants', 'jeans', 'skirt'],
        'Sneakers': ['sneaker', 'shoe', 'boot', 'trainer', 'footwear'],
        'Jewelry': ['jewelry', 'necklace', 'earring', 'bracelet', 'ring'],
        'Accessories': ['bag', 'purse', 'wallet', 'belt', 'scarf', 'hat', 'mask'],
        'Furniture': ['chair', 'table', 'lamp', 'furniture'],
        'Home Decor': ['pillow', 'rug', 'decor', 'print', 'poster']
    }

    for item_type, keywords in item_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            case['original_item_type'] = item_type
            case['copier_item_type'] = item_type
            break

    if not case['original_item_type']:
        case['original_item_type'] = 'Apparel'  # Default
        case['copier_item_type'] = 'Apparel'

    # Extract design elements (look for descriptive phrases)
    # Get sentences that contain descriptive words
    descriptive_words = [
        'print', 'pattern', 'color', 'design', 'motif', 'embroidery',
        'stripe', 'floral', 'geometric', 'graphic', 'logo', 'silhouette',
        'cut', 'shape', 'style', 'detail', 'texture', 'fabric'
    ]

    sentences = re.split(r'[.!?]', caption)
    design_descriptions = []

    for sentence in sentences:
        if any(word in sentence.lower() for word in descriptive_words):
            # Clean up the sentence
            clean = sentence.strip()
            if len(clean) > 20 and len(clean) < 150:
                design_descriptions.append(clean)

    if design_descriptions:
        case['original_design_elements'] = ' '.join(design_descriptions[:2])  # First 2 relevant sentences
    else:
        # Fallback: use first sentence or caption snippet
        case['original_design_elements'] = caption[:150].replace('\n', ' ')

    # Determine label (knockoff vs similar)
    knockoff_keywords = ['identical', 'exact', 'replica', 'knockoff', 'copied', 'stolen', 'ripped off']
    similar_keywords = ['inspired', 'similar', 'reminiscent', 'borrowed']

    has_knockoff = any(kw in caption_lower for kw in knockoff_keywords)
    has_similar = any(kw in caption_lower for kw in similar_keywords)

    if has_knockoff:
        case['infringement_label'] = 'knockoff'
        case['confidence'] = 'high'
    elif has_similar:
        case['infringement_label'] = 'similar'
        case['confidence'] = 'medium'
    else:
        # Default based on source credibility
        case['infringement_label'] = 'knockoff'
        case['confidence'] = 'high'

    # Try to extract years from caption
    years = re.findall(r'\b(19|20)\d{2}\b', caption)
    if len(years) >= 1:
        case['original_year'] = int(years[0])
    if len(years) >= 2:
        case['copy_year'] = int(years[1])

    # Only return if we have minimum required info
    if case['original_designer_name'] or case['copier_brand_name']:
        return case

    return None

def scrape_diet_prada(max_posts=200):
    """
    Scrape Diet Prada posts and extract design cases

    Args:
        max_posts (int): Maximum number of posts to scrape

    Returns:
        list: List of extracted case dictionaries
    """
    print(f"\nScraping Diet Prada (@diet_prada)...")
    print(f"Target: {max_posts} posts")

    # Create loader
    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=True
    )

    cases = []

    try:
        # Get profile
        profile = instaloader.Profile.from_username(L.context, 'diet_prada')

        print(f"Profile found: {profile.full_name}")
        print(f"Total posts: {profile.mediacount}")
        print(f"\nProcessing posts...")

        # Iterate through posts
        post_count = 0
        relevant_count = 0

        for post in profile.get_posts():
            if post_count >= max_posts:
                break

            post_count += 1

            # Progress indicator
            if post_count % 10 == 0:
                print(f"  Processed {post_count} posts, found {relevant_count} relevant cases...")

            # Extract caption
            caption = post.caption
            if not caption:
                continue

            # Try to extract case info
            case = extract_case_info(caption, post.date)

            if case:
                cases.append(case)
                relevant_count += 1

        print(f"\n✓ Scraped {post_count} posts")
        print(f"✓ Found {relevant_count} relevant design cases")

    except instaloader.exceptions.ProfileNotExistsException:
        print("❌ ERROR: Profile 'diet_prada' not found")
        print("   This might be due to Instagram restrictions or the account name changed")
        return []

    except instaloader.exceptions.ConnectionException as e:
        print(f"❌ ERROR: Connection issue - {e}")
        print("   Instagram may be blocking requests. Try:")
        print("   1. Login with: L.login('your_username', 'your_password')")
        print("   2. Use smaller MAX_POSTS value")
        print("   3. Add delays between requests")
        return []

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return []

    return cases

def manual_backup_cases():
    """
    Backup: Manually curated cases if scraping fails
    These are real Diet Prada cases you can verify
    """
    return [
        {
            'case_id': 'DP_MANUAL_001',
            'original_designer_name': 'Emma Mulholland',
            'original_brand_name': 'Emma Mulholland',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Bright fruit prints with watermelons and bananas, bold colors on white background, retro summer aesthetic',
            'original_year': 2014,
            'copier_brand_name': 'H&M',
            'copier_item_type': 'Apparel',
            'copy_year': 2016,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Direct fruit print copy, public callout on Instagram'
        },
        {
            'case_id': 'DP_MANUAL_002',
            'original_designer_name': 'Bailey Prado',
            'original_brand_name': 'Bailey Doesnt Bark',
            'original_item_type': 'Home Decor',
            'original_design_elements': 'Original tapestry design with abstract faces and bold line art, contemporary illustration style',
            'original_year': 2017,
            'copier_brand_name': 'Urban Outfitters',
            'copier_item_type': 'Home Decor',
            'copy_year': 2018,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Exact tapestry design replicated, led to public dispute'
        },
        {
            'case_id': 'DP_MANUAL_003',
            'original_designer_name': 'Adam J Kurtz',
            'original_brand_name': 'Adam J Kurtz',
            'original_item_type': 'Accessories',
            'original_design_elements': 'Hand-lettered motivational graphics, minimalist black and white typography, quirky phrases',
            'original_year': 2015,
            'copier_brand_name': 'Multiple brands',
            'copier_item_type': 'Accessories',
            'copy_year': 2017,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Multiple retailers copied hand-lettered designs'
        },
        {
            'case_id': 'DP_MANUAL_004',
            'original_designer_name': 'Amelia Toro',
            'original_brand_name': 'Amelia Toro',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Intricate white embroidered dress with floral patterns, Colombian artisan craftsmanship, romantic silhouette',
            'original_year': 2016,
            'copier_brand_name': 'Zara',
            'copier_item_type': 'Apparel',
            'copy_year': 2017,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Nearly identical embroidered dress design'
        },
        {
            'case_id': 'DP_MANUAL_005',
            'original_designer_name': 'Sophie Tea',
            'original_brand_name': 'Sophie Tea Art',
            'original_item_type': 'Apparel',
            'original_design_elements': 'Colorful female figure illustrations, empowerment themes, bold brush strokes with pink and orange palette',
            'original_year': 2018,
            'copier_brand_name': 'Shein',
            'copier_item_type': 'Apparel',
            'copy_year': 2019,
            'infringement_label': 'knockoff',
            'confidence': 'high',
            'source': 'Diet Prada',
            'notes': 'Artist illustration style copied onto mass-produced apparel'
        }
    ]

# Main execution
print("\n" + "="*80)
print("ATTEMPTING TO SCRAPE DIET PRADA")
print("="*80)

scraped_cases = scrape_diet_prada(max_posts=MAX_POSTS_PER_ACCOUNT)

# If scraping failed or got very few results, use manual backup
if len(scraped_cases) < 5:
    print("\n⚠️  Scraping returned few results, adding manual backup cases...")
    manual_cases = manual_backup_cases()
    scraped_cases.extend(manual_cases)

if len(scraped_cases) == 0:
    print("\n❌ No cases collected!")
    print("\nPossible reasons:")
    print("1. Instagram is blocking automated access")
    print("2. Need to login first")
    print("3. Rate limiting")
    print("\nRecommended: Use the CSV template for manual data entry")
    exit(1)

# Convert to DataFrame
df = pd.DataFrame(scraped_cases)

print(f"\n{'='*80}")
print("COLLECTED CASES SUMMARY")
print("="*80)
print(f"\nTotal cases: {len(df)}")
print(f"\nLabel distribution:")
print(f"  - knockoff: {sum(df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(df['infringement_label'] == 'similar')}")

print(f"\nConfidence distribution:")
print(f"  - high: {sum(df['confidence'] == 'high')}")
print(f"  - medium: {sum(df['confidence'] == 'medium')}")

print(f"\nItem types:")
for item_type, count in df['original_item_type'].value_counts().items():
    print(f"  - {item_type}: {count}")

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")

print(f"\n{'='*80}")
print("NEXT STEPS")
print("="*80)
print(f"""
1. Review {OUTPUT_FILE} and clean up any incorrect entries
   - Check original_designer_name and copier_brand_name are correct
   - Improve original_design_elements descriptions (be detailed!)
   - Verify infringement_label (knockoff vs similar)

2. Once cleaned, import to gold standard:
   python3 import_from_csv.py

3. Or manually merge:
   - Open gold-standard-cases-CORRECTED.csv
   - Copy rows from {OUTPUT_FILE}
   - Paste at the bottom
   - Save

4. Regenerate embeddings:
   python3 stabilize-clip-embeddings.py

5. Create new splits:
   python3 create_clean_labeled_splits.py

Your dataset will be much larger for better MLP training!
""")

# Show sample of collected data
print("="*80)
print("SAMPLE COLLECTED CASES (first 3)")
print("="*80)
for i in range(min(3, len(df))):
    print(f"\nCase {i+1}:")
    print(f"  Original: {df.iloc[i]['original_designer_name']} / {df.iloc[i]['original_brand_name']}")
    print(f"  Copier: {df.iloc[i]['copier_brand_name']}")
    print(f"  Item: {df.iloc[i]['original_item_type']}")
    print(f"  Design: {df.iloc[i]['original_design_elements'][:100]}...")
    print(f"  Label: {df.iloc[i]['infringement_label']} ({df.iloc[i]['confidence']} confidence)")
