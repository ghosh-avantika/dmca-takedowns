# Guide: Scraping More Design Infringement Cases

## Target Data Format

Your gold standard cases have this structure:
- **case_id**: Unique identifier (e.g., DP_001, DP_002)
- **original_designer_name**: Designer who was copied
- **original_brand_name**: Original brand
- **original_item_type**: Type of item (sneakers, apparel, jewelry, etc.)
- **original_design_elements**: Text description of design features
- **original_year**: Year of original design
- **copier_brand_name**: Who copied it
- **copy_year**: Year of the copy
- **infringement_label**: "knockoff" or "similar"
- **confidence**: "high" or "medium"
- **source**: Where you found this (Diet Prada, Press, Legal filing)
- **notes**: Additional context

## Where to Find Cases

### 1. Diet Prada Instagram (@diet_prada)
- **Best source** for design copying allegations
- Posts show side-by-side comparisons
- Swipe through posts to find "Original vs Copy" content
- Look for captions mentioning brands copying smaller designers

### 2. Fashion News Sites
- **The Fashion Law** (thefashionlaw.com) - Legal cases
- **Business of Fashion** (businessoffashion.com) - Industry news
- **Fashionista** (fashionista.com) - Copy allegations
- **WWD** (wwd.com) - Trade publication

### 3. Legal Databases
- **PACER** (pacer.gov) - US federal court cases
- Search: "design infringement" + "fashion"
- Look for settled cases with clear evidence

### 4. Social Media
- Twitter searches: "copied design", "knockoff", "[brand] copied"
- TikTok fashion commentary accounts
- Reddit r/fashion, r/Designersrights

## Data Collection Template

For each case, collect:

```
Case ID: [Assign unique ID]
Original Designer: [Name]
Original Brand: [Brand name]
Item Type: [apparel/sneakers/jewelry/accessories]
Design Elements: [Describe key visual features - colors, patterns, shapes, motifs]
Original Year: [YYYY]
Copier Brand: [Who copied it]
Copy Year: [YYYY]
Label: [knockoff or similar]
  - knockoff = near-identical, direct copy
  - similar = inspired, borrows key elements
Confidence: [high or medium]
  - high = undeniable similarity
  - medium = noticeable but some differences
Source: [Diet Prada/Press/Legal filing]
Notes: [Context, outcome, any additional info]
```

## Example Workflow

1. **Find a Diet Prada post** showing a copy allegation
2. **Screenshot or note** the comparison images
3. **Research** the original designer and copier brand
4. **Fill out template** with all fields
5. **Add to CSV** using the format script

## Quality Guidelines

- **Focus on clear cases**: Avoid ambiguous "inspired by" situations unless very similar
- **Verify information**: Cross-check with multiple sources when possible
- **Describe design elements well**: This is what CLIP will encode, so be specific
  - Good: "Hand-drawn feminist motifs with pastel pink and mint green palette"
  - Bad: "Nice design"
- **Be consistent with labels**:
  - Knockoff: >90% identical, could confuse consumers
  - Similar: 70-90% similar, clear inspiration but not identical

## Target: 100-200 New Cases

This would give you:
- Current: 105 samples (34 gold + 71 pseudo-labeled)
- With 100 new: 205 samples → ~143 train, 31 val, 31 test
- With 200 new: 305 samples → ~213 train, 46 val, 46 test

Much better for training an MLP classifier!
