"""
Template for scraping Diet Prada posts about design copying
NOTE: Instagram scraping requires authentication and respecting rate limits
This is a TEMPLATE showing the approach - you'll need to adapt it

Alternative: Manually browse Diet Prada and fill out the CSV template
"""

import pandas as pd
import json

print("="*80)
print("DIET PRADA SCRAPING GUIDE")
print("="*80)

print("""
IMPORTANT: Instagram scraping is complex and requires:
1. Instagram account
2. Handling authentication
3. Respecting rate limits
4. Potential account restrictions

RECOMMENDED APPROACH: Manual Collection
========================================

1. Visit: https://www.instagram.com/diet_prada/
2. Look for posts about design copying (usually has comparison images)
3. For each post:
   - Read the caption carefully
   - Note: Original designer/brand
   - Note: Copier brand
   - Note: Design elements described
   - Note: Whether it's a direct copy (knockoff) or inspired (similar)
4. Fill out the template in add_new_cases.py

SEMI-AUTOMATED APPROACH: Export Data
=====================================

Tools that can help export Instagram data:
- Instaloader (Python library): pip install instaloader
- 4K Stogram (Desktop app with export features)
- Phantombuster (Web scraping service with Instagram templates)

These tools can export post captions, dates, and images.
Then manually review each post and categorize.

Example using Instaloader (requires Instagram login):
""")

# Example Instaloader code (commented out - requires setup)
example_code = '''
# Install: pip install instaloader
import instaloader

# Create instance
L = instaloader.Instaloader()

# Login (you'll be prompted for password)
# L.login('your_instagram_username')

# Download Diet Prada posts
# L.download_profile('diet_prada', profile_pic=False)

# This creates a folder with post captions in JSON format
# You can then parse these to extract design copying allegations
'''

print(example_code)

print("""
PARSING DIET PRADA CAPTIONS
===========================

Look for these patterns in captions:
- "Original: [designer/brand]" or "@[designer] original"
- "Copy: [brand]" or "spotted at [brand]"
- "Knockoff alert" (usually means direct copy = knockoff label)
- "Inspired by" (usually means similar label)
- Side-by-side comparisons with "vs" or "Original vs Copy"

Keywords indicating copying:
- "copied", "ripped off", "stolen design", "knockoff"
- "suspiciously similar", "inspired by", "borrowed from"
- "fast fashion brand", "mass retailer copying indie designer"

Example caption structure:
"🚨 @[indie_designer] calls out @[big_brand] for copying their
[item_type] design featuring [design_elements]. The original
from [year] vs the copy from [year]. Thoughts? 👀"
""")

# Helper function to structure data from manual collection
def create_case_template(post_data):
    """
    Helper to create properly formatted case from Diet Prada post

    Args:
        post_data (dict): Post information you collected

    Returns:
        dict: Formatted case ready for add_new_cases.py
    """
    return {
        'case_id': post_data.get('case_id', 'DP_NEW_001'),
        'original_designer_name': post_data.get('original_designer', ''),
        'original_brand_name': post_data.get('original_brand', ''),
        'original_item_type': post_data.get('item_type', 'Apparel'),
        'original_design_elements': post_data.get('design_elements', ''),
        'original_year': post_data.get('original_year', ''),
        'copier_brand_name': post_data.get('copier_brand', ''),
        'copier_item_type': post_data.get('item_type', 'Apparel'),
        'copy_year': post_data.get('copy_year', ''),
        'infringement_label': post_data.get('label', 'knockoff'),
        'confidence': post_data.get('confidence', 'high'),
        'source': 'Diet Prada',
        'notes': post_data.get('notes', '')
    }

# Example of manually collected cases
print("\n" + "="*80)
print("QUICK START: 10 SAMPLE DIET PRADA CASES")
print("="*80)
print("""
Here are some real Diet Prada cases you can look up and add:

1. Tuesday Bassen vs Zara (2016) - Patches
2. Emma Mulholland vs H&M - Fruit prints
3. Indie designer Amelia Toro vs Zara - Embroidered dress
4. Hayden Reis vs Revolve - Knitwear design
5. Bailey Prado vs Urban Outfitters - Tapestry
6. Adam J. Kurtz vs Multiple brands - Graphics
7. People of Print vs Forever 21 - Textile prints
8. Sophie Tea vs Fast fashion - Illustration style
9. Distroller vs Target - Character design
10. Clare V vs Multiple brands - Clutch design

Search Diet Prada for these cases, read the posts, and fill in the template!
""")

print("\n" + "="*80)
print("RECOMMENDED WORKFLOW")
print("="*80)
print("""
Step 1: Browse Diet Prada Instagram
        - Scroll through posts
        - Save/screenshot posts about copying allegations
        - Aim for 20-50 cases per session

Step 2: Create a spreadsheet (or use a notebook)
        - Case ID | Original Designer | Original Brand | Copier |
          Design Elements | Label | Confidence | Notes

Step 3: For each saved post, fill out the spreadsheet row

Step 4: Transfer to add_new_cases.py
        - Copy the template for each case
        - Fill in all fields
        - Run the script to validate and add

Step 5: Regenerate dataset
        - Run stabilize-clip-embeddings.py
        - Run create_clean_labeled_splits.py

Target: 100-200 new cases for a robust dataset!
""")

print("\n✅ Review SCRAPING_GUIDE.md for more detailed instructions")
