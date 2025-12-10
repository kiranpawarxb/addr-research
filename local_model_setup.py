"""
Script to download and test local address parsing model for Indian addresses.

This script will:
1. Download a suitable model from Hugging Face
2. Test it with sample Pune addresses
3. Provide integration guidance
"""

import os
import json
from typing import Dict, Any

def download_model():
    """Download the address parsing model."""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        
        print("Downloading model for Indian address parsing...")
        print("Using: ai4bharat/IndicBERT - optimized for Indian languages and text")
        
        # Using IndicBERT which is trained on Indian data
        model_name = "ai4bharat/indic-bert"
        
        print(f"\nDownloading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Downloading model from {model_name}...")
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        print("\n✓ Model downloaded successfully!")
        print(f"Model saved to: {os.path.expanduser('~/.cache/huggingface/hub')}")
        
        return tokenizer, model
        
    except ImportError:
        print("ERROR: transformers library not installed")
        print("Please run: pip install transformers torch")
        return None, None
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        return None, None


def test_with_pune_addresses(tokenizer, model):
    """Test the model with sample Pune addresses."""
    
    pune_addresses = [
        "Flat 301, Kumar Paradise, Kalyani Nagar, Pune, Maharashtra 411006",
        "A-204, Amanora Park Town, Hadapsar, Pune 411028",
        "Bungalow 12, Koregaon Park, Near Osho Ashram, Pune 411001",
        "Office 505, Cerebrum IT Park, Kumar Road, Pune, Maharashtra 411001",
        "201, Magarpatta City, Hadapsar, Pune 411013"
    ]
    
    print("\n" + "="*70)
    print("TESTING WITH PUNE ADDRESSES")
    print("="*70)
    
    for i, address in enumerate(pune_addresses, 1):
        print(f"\n{i}. Testing: {address}")
        print("-" * 70)
        
        # For now, just show that we can tokenize
        tokens = tokenizer.tokenize(address)
        print(f"   Tokens: {tokens[:10]}...")  # Show first 10 tokens
        

def main():
    """Main function to download and test the model."""
    
    print("="*70)
    print("LOCAL ADDRESS PARSING MODEL SETUP")
    print("="*70)
    print("\nThis script will download a model suitable for Indian address parsing.")
    print("The model will be stored locally and can be used offline.\n")
    
    # Check if transformers is installed
    try:
        import transformers
        import torch
        print(f"✓ transformers version: {transformers.__version__}")
        print(f"✓ torch version: {torch.__version__}")
    except ImportError as e:
        print(f"\n❌ Missing required libraries!")
        print("\nPlease install required packages:")
        print("  pip install transformers torch sentencepiece")
        print("\nFor better performance with CPU:")
        print("  pip install transformers[torch]")
        return
    
    # Download model
    tokenizer, model = download_model()
    
    if tokenizer and model:
        # Test with Pune addresses
        test_with_pune_addresses(tokenizer, model)
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. The model is now downloaded and cached locally")
        print("2. Review the test results above")
        print("3. We'll create a LocalLLMParser class to replace the current LLMParser")
        print("4. The new parser will use this model instead of OpenAI API")
        print("\nModel location:", os.path.expanduser('~/.cache/huggingface/hub'))


if __name__ == "__main__":
    main()
