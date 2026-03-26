# Bash for Data Preprocessing - Practice Guide

## Dataset Overview
You have a sales dataset (`sales_data_raw.csv`) with the following issues:
- Duplicate rows (transaction_id 2 appears twice)
- Inconsistent email formatting (mixed case, extra spaces)
- Missing customer names
- We need to extract useful information and clean it

## Common Bash Commands for Data Preprocessing

### 1. Data Exploration Commands
```bash
# View first few rows
head -n 5 sales_data_raw.csv

# View last few rows
tail -n 5 sales_data_raw.csv

# Count total rows (including header)
wc -l sales_data_raw.csv

# Count unique values in a column (e.g., cities)
cut -d',' -f7 sales_data_raw.csv | tail -n +2 | sort | uniq -c
```

### 2. Filtering Rows
```bash
# Filter rows where product is "Laptop"
head -n 1 sales_data_raw.csv > laptop_sales.csv
grep "Laptop" sales_data_raw.csv >> laptop_sales.csv

# Filter rows where quantity > 2
awk -F',' 'NR==1 || $5 > 2' sales_data_raw.csv

# Remove rows with missing customer names
awk -F',' 'NR==1 || $3 != ""' sales_data_raw.csv
```

### 3. Removing Duplicates
```bash
# Remove duplicate rows based on transaction_id (keep first occurrence)
awk -F',' '!seen[$1]++' sales_data_raw.csv

# Or using sort and uniq (but loses original order)
(head -n 1 sales_data_raw.csv && tail -n +2 sales_data_raw.csv | sort -u)
```

### 4. Column Operations
```bash
# Select specific columns (e.g., date, customer, product, price)
cut -d',' -f2,3,4,6 sales_data_raw.csv

# Reorder columns using awk
awk -F',' 'BEGIN{OFS=","} {print $2,$3,$4,$5,$6}' sales_data_raw.csv

# Add a new column (revenue = quantity * price)
awk -F',' 'BEGIN{OFS=","} NR==1{print $0,"revenue"} NR>1{print $0,$5*$6}' sales_data_raw.csv
```

### 5. Text Cleaning
```bash
# Convert emails to lowercase
awk -F',' 'BEGIN{OFS=","} {$8=tolower($8); print}' sales_data_raw.csv

# Trim whitespace from email column
awk -F',' 'BEGIN{OFS=","} {gsub(/^[ \t]+|[ \t]+$/, "", $8); print}' sales_data_raw.csv

# Replace spaces in customer names with underscores
awk -F',' 'BEGIN{OFS=","} {gsub(/ /, "_", $3); print}' sales_data_raw.csv
```

### 6. Aggregations
```bash
# Count transactions per city
tail -n +2 sales_data_raw.csv | cut -d',' -f7 | sort | uniq -c | sort -rn

# Sum total quantity sold per product
tail -n +2 sales_data_raw.csv | awk -F',' '{sum[$4]+=$5} END {for (p in sum) print p": "sum[p]}'
```

---

## 🎯 EXERCISES - Now Practice!

### Exercise 1: Basic Exploration (Easy)
**Tasks:**
1. Display the header row only
2. Count how many transactions there are (excluding header)
3. List all unique products in the dataset

<details>
<summary>💡 Hints</summary>

- Use `head -n 1` for the header
- Use `wc -l` and subtract 1, or `tail -n +2 | wc -l`
- Use `cut`, `tail`, `sort`, and `uniq`
</details>

### Exercise 2: Data Cleaning (Medium)
**Tasks:**
1. Remove duplicate rows (keep the first occurrence based on transaction_id)
2. Convert all emails to lowercase
3. Trim whitespace from emails
4. Save the cleaned data to `sales_data_clean.csv`

Do all steps in one command pipeline!

<details>
<summary>💡 Hints</summary>

- Use `awk` with `!seen[$1]++` for deduplication
- Use `tolower()` function in awk
- Use `gsub(/^[ \t]+|[ \t]+$/, "", $8)` to trim
- Chain commands with pipes `|`
</details>

### Exercise 3: Filtering & Selection (Medium)
**Tasks:**
1. Create a file `high_value_sales.csv` with only transactions where price >= 100
2. Include only these columns: date, customer_name, product, price
3. Keep the header row

<details>
<summary>💡 Hints</summary>

- Use `awk` with conditions: `$6 >= 100`
- Use `print` to select columns: `print $2,$3,$4,$6`
- Use `NR==1` to always include header
- Remember to set `OFS=","`
</details>

### Exercise 4: Feature Engineering (Hard)
**Tasks:**
1. Add a new column called "revenue" (quantity × price)
2. Add another column called "weekday" by extracting from date (just use "Mon", "Tue", etc. as placeholder - Bash can't parse dates well)
3. Remove rows with missing customer names
4. Save to `sales_data_enhanced.csv`

<details>
<summary>💡 Hints</summary>

- Use `awk` with `BEGIN{OFS=","}` 
- Calculate: `$5*$6` for revenue
- For weekday, just add "Weekday" as a placeholder column
- Use condition `$3 != ""` to filter empty names
</details>

### Exercise 5: Aggregation Report (Advanced)
**Task:** Create a summary report showing:
- Total number of transactions per city
- Sorted from highest to lowest
- Output format: "CityName: Count"

<details>
<summary>💡 Hints</summary>

- Extract city column: `cut -d',' -f7`
- Skip header: `tail -n +2`
- Count occurrences: `sort | uniq -c`
- Format and sort: `sort -rn` then `awk` for formatting
</details>

---

## ✅ Solutions

Run these commands after attempting the exercises yourself!

### Solution 1:
```bash
# 1. Header only
head -n 1 sales_data_raw.csv

# 2. Count transactions (excluding header)
tail -n +2 sales_data_raw.csv | wc -l

# 3. Unique products
cut -d',' -f4 sales_data_raw.csv | tail -n +2 | sort -u
```

### Solution 2:
```bash
awk -F',' 'BEGIN{OFS=","} !seen[$1]++ {$8=tolower($8); gsub(/^[ \t]+|[ \t]+$/, "", $8); print}' sales_data_raw.csv > sales_data_clean.csv
```

### Solution 3:
```bash
awk -F',' 'BEGIN{OFS=","} NR==1 {print $2,$3,$4,$6} NR>1 && $6>=100 {print $2,$3,$4,$6}' sales_data_raw.csv > high_value_sales.csv
```

### Solution 4:
```bash
awk -F',' 'BEGIN{OFS=","} NR==1 {print $0,"revenue","weekday"} NR>1 && $3!="" {print $0,$5*$6,"Weekday"}' sales_data_raw.csv > sales_data_enhanced.csv
```

### Solution 5:
```bash
echo "City Transaction Report:"
tail -n +2 sales_data_raw.csv | cut -d',' -f7 | sort | uniq -c | sort -rn | awk '{print $2": "$1}'
```

---

## 🎓 When to Use Bash vs Python/R

### Use Bash when:
✅ Quick data exploration (row counts, unique values)
✅ Simple row filtering (by patterns or simple conditions)
✅ Removing duplicates
✅ Selecting/reordering columns
✅ Basic text cleaning (case conversion, trimming)
✅ Working with very large files (Bash streams, doesn't load into memory)
✅ File splitting/merging/sampling

### Use Python/Pandas or R when:
❌ Complex feature engineering (like city → lat/long you mentioned)
❌ Handling missing data with sophisticated methods
❌ Statistical transformations (scaling, normalization)
❌ Date/time parsing and manipulation
❌ Joining multiple datasets
❌ Any operation requiring "looking up" other data
❌ Machine learning preprocessing (encoding, scaling, etc.)

## Real-World Workflow

In practice, data scientists often use **both**:

1. **Bash** for initial cleanup: removing duplicates, filtering obvious bad rows, sampling large files
2. **Python/R** for actual feature engineering: encoding categories, handling missing values, creating ML features

Example:
```bash
# Step 1: Quick Bash preprocessing (10GB file)
cat huge_dataset.csv | 
  awk '!seen[$1]++' |           # Remove duplicates
  grep -v "^,,,,,," |           # Remove completely empty rows
  head -n 1000000 > sample.csv  # Take first 1M rows

# Step 2: Python for actual ML preprocessing
# (encoding cities, handling nulls, feature engineering)
```

This is why your senior said Bash is "faster" - for large files, Bash processes line-by-line without loading everything into memory, making initial cleaning very efficient!
