"""
Investigate and fix the gold standard CSV label issue
"""

import pandas as pd

print("="*80)
print("INVESTIGATING GOLD STANDARD CSV LABELS")
print("="*80)

# Load the CSV
df = pd.read_csv('gold-standard-cases.csv')
df_clean = df.dropna(how='all')

print(f"\nTotal rows: {len(df_clean)}")
print(f"\nColumns: {df_clean.columns.tolist()}")

# Display first few rows to understand structure
print("\n" + "="*80)
print("FIRST 10 ROWS - KEY COLUMNS")
print("="*80)

for idx in range(min(10, len(df_clean))):
    row = df_clean.iloc[idx]
    print(f"\nRow {idx+1}:")
    print(f"  case_id: {row.get('case_id', 'N/A')}")
    print(f"  infringement_label: {row.get('infringement_label', 'N/A')}")
    print(f"  confidence: {row.get('confidence', 'N/A')}")
    print(f"  design_elements: {str(row.get('original_design_elements', 'N/A'))[:60]}")

# Check all unique values in relevant columns
print("\n" + "="*80)
print("UNIQUE VALUES IN KEY COLUMNS")
print("="*80)

if 'infringement_label' in df_clean.columns:
    print(f"\ninfringement_label column values:")
    print(df_clean['infringement_label'].value_counts())

if 'confidence' in df_clean.columns:
    print(f"\nconfidence column values:")
    print(df_clean['confidence'].value_counts())

if 'label' in df_clean.columns:
    print(f"\nlabel column values:")
    print(df_clean['label'].value_counts())

# Analysis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("\n🔍 Analysis:")
print("   The 'infringement_label' column contains:")
print("   - 'knockoff' (7) - ✓ Valid infringement label")
print("   - 'similar' (3) - ✓ Valid infringement label")
print("   - 'high' (11) - ❌ Looks like a confidence level")
print("   - 'medium' (9) - ❌ Looks like a confidence level")
print("   - 'confidence' (4) - ❌ Corrupted/misplaced")

print("\n💡 Hypothesis:")
print("   It appears the 'confidence' column values may have been")
print("   accidentally placed in the 'infringement_label' column,")
print("   OR the actual labels are in a different column.")

# Check if there's a pattern
print("\n" + "="*80)
print("CHECKING FOR PATTERNS")
print("="*80)

# Group by what's in infringement_label and see confidence values
if 'confidence' in df_clean.columns and 'infringement_label' in df_clean.columns:
    print("\nCross-tabulation: infringement_label vs confidence:")
    crosstab = pd.crosstab(df_clean['infringement_label'],
                           df_clean['confidence'],
                           margins=True,
                           dropna=False)
    print(crosstab)

# Proposed fix
print("\n" + "="*80)
print("PROPOSED FIX")
print("="*80)

print("\nOption 1: Use 'confidence' column as the actual label")
print("   If high/medium confidence rows should all be 'knockoff' or 'similar'")

print("\nOption 2: Infer from notes or source")
print("   Check if 'notes' column has clues about actual infringement type")

print("\nOption 3: Manual correction needed")
print("   Review the PDF or source data to get correct labels")

# Check notes column for clues
if 'notes' in df_clean.columns:
    print("\n" + "="*80)
    print("CHECKING 'notes' COLUMN FOR CLUES")
    print("="*80)

    for idx in range(min(15, len(df_clean))):
        row = df_clean.iloc[idx]
        label = row.get('infringement_label', 'N/A')
        notes = str(row.get('notes', ''))

        if label in ['high', 'medium', 'confidence']:
            print(f"\nRow {idx+1} - Label: '{label}'")
            print(f"  Notes: {notes[:80]}")

            # Check for keywords
            notes_lower = notes.lower()
            if 'identical' in notes_lower or 'exact' in notes_lower or 'replicated' in notes_lower:
                print("  → Suggests: KNOCKOFF")
            elif 'similar' in notes_lower or 'inspired' in notes_lower:
                print("  → Suggests: SIMILAR")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nBased on the notes analysis, I can create a corrected CSV.")
print("Would you like me to:")
print("  1. Auto-fix based on notes keywords")
print("  2. Show you each case for manual review")
print("  3. Use confidence column to map to labels")
