# Parser Comparison Report: Local Rule-Based vs Shiprocket

**Generated:** 2025-12-09 22:46:55

## Executive Summary

**Recommended Parser:** Local Rule-Based

## Dataset

- Total Addresses Tested: 20

## Performance Comparison

| Metric | Local Rule-Based | Shiprocket ML-Based |
|--------|------------------|---------------------|
| Success Rate | 100.0% | 40.0% |
| Total Time | 0.01s | 17.79s |
| Avg Time/Address | 0.45ms | 889.54ms |
| Speed Ratio | 1.0x | 1976.6x |

## Field Extraction Rates

| Field | Local | Shiprocket | Agreement |
|-------|-------|------------|----------|
| unit_number | 45.0% | 35.0% | 25.0% |
| society_name | 15.0% | 35.0% | 12.5% |
| landmark | 25.0% | 25.0% | 37.5% |
| road | 0.0% | 20.0% | 50.0% |
| sub_locality | 5.0% | 5.0% | 87.5% |
| locality | 10.0% | 35.0% | 25.0% |
| city | 40.0% | 15.0% | 75.0% |
| district | 40.0% | 15.0% | 75.0% |
| state | 10.0% | 0.0% | 87.5% |
| country | 100.0% | 40.0% | 100.0% |
| pin_code | 10.0% | 5.0% | 100.0% |

**Average Agreement Rate:** 61.4%

## Detailed Analysis

### Local Rule-Based Parser

**Strengths:**
- Extremely fast (<1ms per address)
- No model downloads or dependencies
- Instant startup time
- Low memory footprint
- Works offline

**Weaknesses:**
- May struggle with highly unstructured addresses
- Limited adaptability to new patterns

### Shiprocket ML-Based Parser

**Strengths:**
- Specifically trained for Indian addresses
- Uses fine-tuned IndicBERT model
- Better handling of variations
- Can improve with additional training

**Weaknesses:**
- Slower processing (ML inference)
- Requires ~500MB model download
- Higher memory requirements
- Longer startup time

## Recommendation

**Use Local Rule-Based Parser** for:
- High-volume processing where speed matters
- Production deployments with well-formatted addresses
- Resource-constrained environments
- When instant startup is required

**Consider Shiprocket** if:
- You need better handling of unstructured addresses
- You have GPU infrastructure available
- Processing speed is not critical

## Files Generated

- `local_shiprocket_comparison.csv` - Detailed row-by-row comparison
- `LOCAL_SHIPROCKET_COMPARISON.md` - This summary report
