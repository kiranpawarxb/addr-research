# Parser Comparison Results - Quick Start Guide

## 🎯 Bottom Line

**USE THE LOCAL RULE-BASED PARSER**

It's 4,685x faster, has 100% success rate, and better field extraction than Shiprocket.

---

## 📊 Test Results Summary

| Metric | Local | Shiprocket | Winner |
|--------|-------|------------|--------|
| Success Rate | 100% | 89% | **Local** |
| Speed | 0.17ms | 796ms | **Local** (4,685x faster) |
| Field Quality | 31.3% | 28.0% | **Local** |
| Setup | None | 500MB download | **Local** |
| Failures | 0 | 11 out of 100 | **Local** |

---

## 📁 Generated Files

### 1. **COMPARISON_RESULTS_SUMMARY.txt** ⭐ START HERE
Quick visual summary with charts and key metrics. Perfect for a fast overview.

### 2. **PARSER_COMPARISON_DECISION_SUMMARY.md** 📖 DETAILED GUIDE
Comprehensive analysis with:
- Detailed metrics breakdown
- Field-by-field comparison
- Scalability analysis
- Sample results
- Implementation guide

### 3. **LOCAL_SHIPROCKET_COMPARISON.md** 📋 FULL REPORT
Complete comparison report with all statistics and recommendations.

### 4. **local_shiprocket_comparison.csv** 📊 RAW DATA
Row-by-row results for all 100 test addresses. Use this to:
- Analyze specific address patterns
- Identify where each parser excels
- Validate the comparison results

### 5. **compare_local_shiprocket.py** 🔧 COMPARISON TOOL
The script used to run the comparison. You can:
- Re-run with different parameters
- Test with your own address samples
- Adjust the sample size

---

## 🚀 Quick Start

### View Results
```bash
# Quick summary (recommended first read)
type COMPARISON_RESULTS_SUMMARY.txt

# Detailed decision guide
type PARSER_COMPARISON_DECISION_SUMMARY.md

# Full report
type LOCAL_SHIPROCKET_COMPARISON.md

# Raw data (open in Excel/spreadsheet)
start local_shiprocket_comparison.csv
```

### Re-run Comparison
```bash
# Test with 100 addresses (default)
python compare_local_shiprocket.py

# Test with 50 addresses
python compare_local_shiprocket.py --limit 50

# Test with different CSV
python compare_local_shiprocket.py --csv your_file.csv --column address_column

# Use GPU for Shiprocket (if available)
python compare_local_shiprocket.py --gpu
```

---

## 📈 Key Findings

### Performance
- **Local:** 0.02 seconds for 100 addresses
- **Shiprocket:** 79.66 seconds for 100 addresses
- **Difference:** Shiprocket is 4,685x slower

### Reliability
- **Local:** 100% success rate (0 failures)
- **Shiprocket:** 89% success rate (11 failures)

### Scalability
For your full 53MB dataset (~500,000 addresses):
- **Local:** ~1.4 minutes ✅
- **Shiprocket:** ~110 hours (4.6 days) ❌

### Field Extraction
Local parser wins in 9 out of 11 fields:
- ✅ unit_number, society_name, landmark
- ✅ sub_locality, city, district
- ✅ state, country, pin_code

Shiprocket wins in 2 fields:
- ✅ road, locality

---

## 💡 Decision Factors

### Choose Local Parser If:
✅ You need fast processing (production use)  
✅ You want 100% reliability  
✅ You're processing large volumes  
✅ You want zero setup/dependencies  
✅ You need offline capability  
✅ You want instant startup  

### Choose Shiprocket Only If:
⚠️ You specifically need better road/locality extraction  
⚠️ You can tolerate 11% failure rate  
⚠️ You have GPU infrastructure  
⚠️ You're processing <10 addresses at a time  
⚠️ Speed doesn't matter  

---

## 🔧 Implementation

### Update Your Config
```yaml
# config/config.yaml
llm:
  parser_type: "local"  # Change from "shiprocket" to "local"
```

### Run Your Pipeline
```bash
python -m src --input export_customer_address_store_p0.csv --output results.csv
```

---

## 📊 Sample Results

### Example 1: Simple Address
**Input:** `"flat-302, friendship residency, veerbhadra nagar road"`

| Parser | Result |
|--------|--------|
| Local | ✅ Extracted: unit=302, society=friendship residency |
| Shiprocket | ❌ Failed to parse |

### Example 2: Complex Address
**Input:** `"124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir"`

| Parser | Result |
|--------|--------|
| Local | ✅ Extracted: unit=124/1, landmark=datta mandir, locality=sadgurunagar, city=Pune |
| Shiprocket | ❌ Failed to parse |

### Example 3: Where Both Succeeded
**Input:** `"panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503"`

| Parser | Result |
|--------|--------|
| Local | ✅ Parsed successfully |
| Shiprocket | ✅ Parsed successfully (better road extraction) |

---

## ⚠️ Shiprocket Issues Observed

During testing, Shiprocket had:
1. **Model loading errors** - transformers library issues
2. **Tensor device errors** - "meta tensors" problems
3. **11% failure rate** - 11 out of 100 addresses failed
4. **Extremely slow** - 796ms per address vs 0.17ms
5. **Inconsistent behavior** - parallel processing issues

---

## 📞 Need Help?

### Questions About Results
- Check the detailed CSV for specific examples
- Review PARSER_COMPARISON_DECISION_SUMMARY.md
- Look at sample results in the summary files

### Want to Re-test
```bash
# Run comparison with your own parameters
python compare_local_shiprocket.py --help
```

### Implementation Issues
- Verify config.yaml has `parser_type: "local"`
- Check that src/local_llm_parser.py exists
- Run: `python test_local_parser_pune.py` to verify setup

---

## 📝 Test Details

- **Date:** December 9, 2025
- **Dataset:** export_customer_address_store_p0.csv
- **Sample Size:** 100 addresses
- **Full Dataset:** 53MB (~500,000 addresses)
- **Test Environment:** Windows, Python 3.9, CPU only

---

## 🎯 Final Recommendation

**Use Local Rule-Based Parser for production.**

It's faster, more reliable, and better suited for your dataset size. The comparison results clearly show it's the right choice for high-volume address processing.

---

**Generated by:** compare_local_shiprocket.py  
**Last Updated:** December 9, 2025
