import pandas as pd

# Load the raw, untouched file
df = pd.read_csv('lk_telecom_synthetic_100k.csv')

print("==========================================================")
print("     STEP 0: DATA ENGINEER DIAGNOSTIC AUDIT REPORT        ")
print("==========================================================\n")

# QUESTION 1: How many rows are there, and are IDs unique?
total_rows = len(df)
unique_ids = df['customer_id'].nunique()
exact_dups = df.duplicated().sum()
dup_ids = total_rows - unique_ids
hidden_dups = dup_ids - exact_dups

print(f"1. PRIMARY KEY & DUPLICATE AUDIT:")
print(f"   - Total Rows:                   {total_rows:,}")
print(f"   - Unique customer_ids:          {unique_ids:,}")
print(f"   - Duplicate customer_ids:       {dup_ids:,}")
print(f"   - Exact 100% Duplicate Rows:    {exact_dups:,}")
print(f"   - Hidden Duplicates (Diffs):    {hidden_dups:,}\n")

# QUESTION 2: Are there any useless columns where every row has the same value?
dead_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
print(f"2. ZERO-VARIANCE COLUMNS (Drop Immediately): {dead_cols}\n")

# QUESTION 3: Which text columns have uppercase/lowercase typos or unsorted pipes?
print("3. CATEGORICAL CASING & FORMATTING ISSUES:")
for col in df.select_dtypes(include='object').columns:
    if col not in ['customer_id', 'complaint_text']:
        raw_u = df[col].dropna().nunique()
        clean_u = df[col].dropna().astype(str).str.strip().str.lower().nunique()
        if raw_u != clean_u:
            print(f"   - [CASE MISMATCH] '{col}': {raw_u} raw values collapse to {clean_u} lowercase values")
        
        # Check if pipe-separated values (like 'A|B' vs 'B|A') exist
        if df[col].dropna().astype(str).str.contains(r'\|').any():
            sorted_u = df[col].dropna().astype(str).apply(lambda x: '|'.join(sorted(x.split('|')))).nunique()
            if raw_u != sorted_u:
                print(f"   - [DELIMITER ORDER MISMATCH] '{col}': {raw_u} raw values collapse to {sorted_u} sorted values")

# QUESTION 4: Which columns are missing the exact same number of values?
print("\n4. MISSING VALUE PATTERNS (Grouped by Exact Missing Count):")
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values()
for count_val, group in null_counts.groupby(null_counts):
    pct = (count_val / total_rows) * 100
    print(f"   - Exactly {count_val:,} missing ({pct:.2f}%): {list(group.index)}")