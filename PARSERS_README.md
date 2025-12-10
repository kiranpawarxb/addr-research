# Address Parser Options

## Overview

The Address Consolidation System supports **three different parsers** for extracting structured address components:

## 1. Rule-Based Local Parser ⚡

**Fast, lightweight, rule-based extraction**

- **Speed**: <1ms per address
- **Cost**: Free
- **Setup**: Instant (no downloads)
- **Offline**: Yes
- **Best for**: Well-formatted Indian addresses, production use

```yaml
# config.yaml
llm:
  parser_type: "local"
```

## 2. IndicBERT Parser 🤖

**ML-based transformer model for Indian text**

- **Speed**: ~200-500ms per address (CPU), ~50-100ms (GPU)
- **Cost**: Free
- **Setup**: ~500MB model download on first run
- **Offline**: Yes
- **GPU Support**: Yes (optional)
- **Best for**: Complex/unstructured addresses, ML-based extraction

```yaml
# config.yaml
llm:
  parser_type: "indicbert"
  use_gpu: false  # Set true for GPU acceleration
```

**Installation:**
```bash
pip install transformers torch
```

## 3. OpenAI Parser ☁️

**Cloud-based GPT models (original implementation)**

- **Speed**: ~1-3 seconds per address (network dependent)
- **Cost**: ~$0.01-0.03 per 1,000 addresses
- **Setup**: API key required
- **Offline**: No
- **Best for**: International addresses, highest accuracy needs

```yaml
# config.yaml
llm:
  parser_type: "openai"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
```

## Quick Comparison

| Feature | Rule-Based | IndicBERT | OpenAI |
|---------|-----------|-----------|---------|
| Speed | ⚡ Very Fast | 🐌 Slower | 🐌 Slowest |
| Cost | 💰 Free | 💰 Free | 💸 Paid |
| Offline | ✅ Yes | ✅ Yes | ❌ No |
| Setup | ✅ Instant | ⏳ 500MB | ✅ API key |
| Accuracy | ✅ Excellent | ✅ Excellent | ✅ Excellent |

## Comparing Parsers

### Quick Comparison (3 addresses)
```bash
python quick_compare.py
```

### Full Comparison (10 addresses with metrics)
```bash
python compare_parsers.py
```

Output: `parser_comparison.csv` with detailed field-by-field comparison

## Recommendations

### Use Rule-Based When:
- ✅ Processing well-formatted addresses
- ✅ Speed is critical
- ✅ High-volume processing
- ✅ Want zero dependencies

### Use IndicBERT When:
- ✅ Dealing with unstructured addresses
- ✅ Want ML-based extraction
- ✅ Have GPU available
- ✅ Need better variation handling

### Use OpenAI When:
- ✅ Processing international addresses
- ✅ Need highest accuracy
- ✅ Small volumes
- ✅ Budget allows

## Documentation

- **Full Comparison Guide**: `docs/parser_comparison_guide.md`
- **Local Parser Guide**: `docs/local_parser_guide.md`
- **Setup Complete**: `PARSER_OPTIONS_COMPLETE.md`

## Quick Start

1. **Choose parser** in `config/config.yaml`
2. **Run comparison** (optional): `python compare_parsers.py`
3. **Process addresses**: `python -m src --input data.csv --output results.csv`

That's it! The system will use your chosen parser automatically.
