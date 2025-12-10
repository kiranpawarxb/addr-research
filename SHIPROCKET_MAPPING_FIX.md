# Shiprocket Parser Entity Mapping Fix

## Problem Identified

The Shiprocket IndicBERT model was extracting 0 society names because the entity label mapping was incorrect.

## Root Cause

The Shiprocket model uses **different entity labels** than initially assumed:

### Actual Labels Used by Shiprocket Model:
- `building_name` - for society/building names
- `house_details` - for unit/flat numbers  
- `landmarks` - for landmarks/nearby places
- `locality` - for locality/area names
- `city` - for city names
- `street` - for street/road names (when present)
- `state` - for state names (when present)
- `pincode` - for PIN codes (when present)

### Original (Incorrect) Mapping:
The code was looking for labels like:
- `BUILDING`, `SOCIETY`, `COMPLEX` (uppercase, different names)
- `HOUSE_NUMBER`, `FLAT`, `UNIT`
- `LANDMARK`, `NEAR`

## Solution Applied

Updated `src/shiprocket_parser.py` to use the correct entity labels:

```python
# Map Shiprocket entity types to our fields
if entity_type in ['house_details', 'house_number', 'flat', 'unit']:
    if not fields['unit_number']:
        fields['unit_number'] = entity_text
elif entity_type in ['building_name', 'building', 'society', 'complex']:
    if not fields['society_name']:
        fields['society_name'] = entity_text
elif entity_type in ['landmarks', 'landmark', 'near']:
    if not fields['landmark']:
        fields['landmark'] = entity_text
# ... etc
```

### Key Changes:
1. Changed to **lowercase** entity type matching
2. Added **trailing comma cleanup** for extracted text
3. Mapped `building_name` → `society_name`
4. Mapped `house_details` → `unit_number`
5. Mapped `landmarks` → `landmark`

## Results

### Before Fix:
- **Society Names Extracted:** 0 out of 100 (0%)
- **Unit Numbers Extracted:** 0 out of 100 (0%)
- **Landmarks Extracted:** 0 out of 100 (0%)

### After Fix (20 address sample):
- **Society Names Extracted:** 7 out of 20 (35%)
- **Unit Numbers Extracted:** 7 out of 20 (35%)
- **Landmarks Extracted:** 5 out of 20 (25%)

### Improvement:
- ✅ **Society name extraction improved from 0% to 35%**
- ✅ **Unit number extraction improved from 0% to 35%**
- ✅ **Landmark extraction improved from 0% to 25%**

## Test Examples

### Example 1: "flat-302, friendship residency, veerbhadra nagar road"
**Before Fix:**
- Society: '' (empty)
- Unit: '' (empty)

**After Fix:**
- Society: 'flat-302, friendship residency' ✓
- Unit: '' (model grouped it with building)
- Locality: 'veerbhadra nagar road' ✓

### Example 2: "506, amnora chembers, east amnora town center, hadapsar, pune"
**Before Fix:**
- Society: '' (empty)
- Unit: '' (empty)

**After Fix:**
- Society: 'amnora chembers' ✓
- Unit: '506' ✓
- Landmark: 'east amnora town center' ✓
- Locality: 'hadapsar' ✓
- City: 'pune' ✓

### Example 3: "101, shivam building, behind shree kalyani nursing home, lohegaon, pune"
**Before Fix:**
- Society: '' (empty)
- Unit: '' (empty)

**After Fix:**
- Society: 'shivam building' ✓
- Unit: '101' ✓
- Landmark: 'behind shree kalyani nursing home' ✓
- Locality: 'lohegaon' ✓
- City: 'pune' ✓

## Remaining Issues

Despite the fix, Shiprocket still has significant problems:

1. **Low Success Rate:** 40% (8/20) vs Local's 100% (20/20)
2. **Slow Performance:** 889ms per address vs Local's 0.45ms (1,976x slower)
3. **Tensor Errors:** Still encountering "meta tensors" errors causing 60% failure rate
4. **Inconsistent Results:** Parallel processing causes device switching issues

## Recommendation

Even with the fixed mapping, **Local Rule-Based Parser remains the better choice**:

| Metric | Local | Shiprocket (Fixed) |
|--------|-------|-------------------|
| Success Rate | 100% | 40% |
| Speed | 0.45ms | 889ms |
| Society Extraction | 15% | 35% |
| Overall Reliability | ✅ Excellent | ⚠️ Poor |

### Why Local is Still Better:
1. **100% success rate** (no failures)
2. **1,976x faster** processing
3. **No model loading errors** or tensor issues
4. **Consistent, predictable** behavior
5. **Production-ready** for high-volume processing

### When to Use Shiprocket (Fixed):
- Small batches (<10 addresses)
- When you specifically need better locality extraction
- When you have GPU available
- When you can tolerate 60% failure rate
- When speed is not critical

## Files Modified

- `src/shiprocket_parser.py` - Fixed entity label mapping
- `debug_shiprocket_labels.py` - Debug script to discover actual labels
- `test_shiprocket_fix.py` - Test script to verify the fix

## Testing

To verify the fix works:
```bash
python test_shiprocket_fix.py
```

To run a new comparison:
```bash
python compare_local_shiprocket.py --limit 20
```

---

**Date:** December 9, 2025  
**Status:** ✅ Fixed - Entity mapping corrected  
**Impact:** Shiprocket now extracts society names, but Local parser still recommended for production
