#!/bin/bash

# Bash Preprocessing Practice - Solution Checker
# This script helps you verify your exercise solutions

echo "================================================"
echo "   Bash Data Preprocessing - Exercise Checker"
echo "================================================"
echo ""

# Check if raw data file exists
if [ ! -f "sales_data_raw.csv" ]; then
    echo "❌ Error: sales_data_raw.csv not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

echo "✅ Found sales_data_raw.csv"
echo ""

# Function to check if a file exists and has content
check_file() {
    local filename=$1
    local exercise=$2
    
    if [ -f "$filename" ]; then
        local lines=$(wc -l < "$filename")
        echo "✅ $filename exists ($lines lines)"
        return 0
    else
        echo "❌ $filename not found (Exercise $exercise)"
        return 1
    fi
}

# Exercise 1 Solutions
echo "=== Exercise 1: Basic Exploration ==="
echo ""
echo "Correct Answers:"
echo "1. Header row:"
head -n 1 sales_data_raw.csv
echo ""
echo "2. Number of transactions (excluding header):"
tail -n +2 sales_data_raw.csv | wc -l
echo ""
echo "3. Unique products:"
cut -d',' -f4 sales_data_raw.csv | tail -n +2 | sort -u
echo ""
echo "Press Enter to continue..."
read

# Exercise 2 Check
echo ""
echo "=== Exercise 2: Data Cleaning ==="
echo ""
if check_file "sales_data_clean.csv" "2"; then
    echo ""
    echo "Your cleaned file preview:"
    head -n 5 sales_data_clean.csv
    echo ""
    
    # Check for duplicates
    dup_count=$(tail -n +2 sales_data_clean.csv | cut -d',' -f1 | sort | uniq -d | wc -l)
    if [ $dup_count -eq 0 ]; then
        echo "✅ No duplicate transaction IDs found"
    else
        echo "⚠️  Warning: Found $dup_count duplicate transaction IDs"
    fi
    
    # Check for uppercase emails
    upper_count=$(tail -n +2 sales_data_clean.csv | cut -d',' -f8 | grep '[A-Z]' | wc -l)
    if [ $upper_count -eq 0 ]; then
        echo "✅ All emails are lowercase"
    else
        echo "⚠️  Warning: Found $upper_count emails with uppercase letters"
    fi
else
    echo ""
    echo "Expected solution:"
    echo "awk -F',' 'BEGIN{OFS=\",\"} !seen[\$1]++ {\$8=tolower(\$8); gsub(/^[ \\t]+|[ \\t]+\$/, \"\", \$8); print}' sales_data_raw.csv > sales_data_clean.csv"
fi
echo ""
echo "Press Enter to continue..."
read

# Exercise 3 Check
echo ""
echo "=== Exercise 3: Filtering & Selection ==="
echo ""
if check_file "high_value_sales.csv" "3"; then
    echo ""
    echo "Your filtered file preview:"
    head -n 5 high_value_sales.csv
    echo ""
    
    # Check column count
    col_count=$(head -n 1 high_value_sales.csv | awk -F',' '{print NF}')
    if [ $col_count -eq 4 ]; then
        echo "✅ Correct number of columns (4)"
    else
        echo "⚠️  Warning: Expected 4 columns, found $col_count"
    fi
    
    # Check if all prices >= 100
    low_price=$(tail -n +2 high_value_sales.csv | cut -d',' -f4 | awk '$1 < 100 {print $1}' | wc -l)
    if [ $low_price -eq 0 ]; then
        echo "✅ All prices are >= 100"
    else
        echo "⚠️  Warning: Found $low_price rows with price < 100"
    fi
else
    echo ""
    echo "Expected solution:"
    echo "awk -F',' 'BEGIN{OFS=\",\"} NR==1 {print \$2,\$3,\$4,\$6} NR>1 && \$6>=100 {print \$2,\$3,\$4,\$6}' sales_data_raw.csv > high_value_sales.csv"
fi
echo ""
echo "Press Enter to continue..."
read

# Exercise 4 Check
echo ""
echo "=== Exercise 4: Feature Engineering ==="
echo ""
if check_file "sales_data_enhanced.csv" "4"; then
    echo ""
    echo "Your enhanced file preview:"
    head -n 5 sales_data_enhanced.csv
    echo ""
    
    # Check for new columns
    col_count=$(head -n 1 sales_data_enhanced.csv | awk -F',' '{print NF}')
    if [ $col_count -eq 10 ]; then
        echo "✅ Correct number of columns (10 = 8 original + 2 new)"
    else
        echo "⚠️  Expected 10 columns, found $col_count"
    fi
    
    # Check for empty customer names
    empty_names=$(tail -n +2 sales_data_enhanced.csv | cut -d',' -f3 | grep -c '^$')
    if [ $empty_names -eq 0 ]; then
        echo "✅ No empty customer names"
    else
        echo "⚠️  Warning: Found $empty_names rows with empty customer names"
    fi
else
    echo ""
    echo "Expected solution:"
    echo "awk -F',' 'BEGIN{OFS=\",\"} NR==1 {print \$0,\"revenue\",\"weekday\"} NR>1 && \$3!=\"\" {print \$0,\$5*\$6,\"Weekday\"}' sales_data_raw.csv > sales_data_enhanced.csv"
fi
echo ""
echo "Press Enter to continue..."
read

# Exercise 5
echo ""
echo "=== Exercise 5: Aggregation Report ==="
echo ""
echo "Expected output:"
echo "City Transaction Report:"
tail -n +2 sales_data_raw.csv | cut -d',' -f7 | sort | uniq -c | sort -rn | awk '{print $2": "$1}'
echo ""

echo ""
echo "================================================"
echo "             Practice Complete! 🎉"
echo "================================================"
echo ""
echo "Try running the commands yourself and compare with the solutions above."
echo ""
echo "Remember: Bash is great for simple, fast operations on large files,"
echo "but use Python/R for complex transformations like:"
echo "  - City → Latitude/Longitude (requires lookup)"
echo "  - Date parsing and feature extraction"
echo "  - Statistical scaling/normalization"
echo "  - One-hot encoding or other ML preprocessing"
echo ""
