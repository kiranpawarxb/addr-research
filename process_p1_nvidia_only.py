#!/usr/bin/env python3
"""
Process export_customer_address_store_p1.csv using NVIDIA GPU ONLY:
- Single NVIDIA GPU processing
- Limited to 10k addresses for testing
- Simple, reliable implementation
"""

import sys
import os
import pandas as pd
import logging
import time
from datetime import datetime
from typing import List, Dict
import re

# Add src to path
sys.path.insert(0, 'src')

from src.models import ParsedAddress


class NvidiaOnlyParser:
    """Simple NVIDIA GPU-only parser."""
    
    def __init__(self, model_name: str = "shiprocket-ai/open-indicbert-indian-address-ner", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = None
        
    def setup_nvidia_gpu(self):
        """Set up NVIDIA GPU processing."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            if not torch.cuda.is_available():
                print("❌ NVIDIA CUDA not available")
                return False
            
            print("🔧 Setting up NVIDIA GPU processing...")
            print(f"   CUDA devices available: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
            
            # Load model on NVIDIA GPU
            print("   Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print("   Loading model...")
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision for speed
                device_map=None
            ).cuda()
            
            print("   Creating pipeline...")
            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=0,  # CUDA device 0
                aggregation_strategy="simple",
                batch_size=self.batch_size
            )
            
            print("✅ NVIDIA GPU setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup NVIDIA GPU: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def parse_addresses(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse addresses using NVIDIA GPU."""
        if not addresses:
            return []
        
        if not self._pipeline:
            print("❌ GPU not initialized")
            return [ParsedAddress(parse_success=False, parse_error="GPU not initialized") for _ in addresses]
        
        start_time = time.time()
        print(f"🚀 Processing {len(addresses)} addresses on NVIDIA GPU...")
        
        results = []
        
        try:
            # Clean addresses
            print("   Cleaning address texts...")
            batch_texts = []
            for addr in addresses:
                cleaned = self._clean_address_text(addr)
                batch_texts.append(cleaned)
            
            # Process in batches
            print(f"   Processing in batches of {self.batch_size}...")
            total_batches = (len(batch_texts) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(batch_texts), self.batch_size):
                batch_num = (i // self.batch_size) + 1
                batch = batch_texts[i:i + self.batch_size]
                
                if batch_num % 10 == 0 or batch_num == total_batches:
                    print(f"   Processing batch {batch_num}/{total_batches}...")
                
                # Process batch
                entities_batch = self._pipeline(batch)
                
                # Extract fields for each address in batch
                for j, (addr, entities) in enumerate(zip(addresses[i:i + self.batch_size], entities_batch)):
                    parsed = self._extract_fields_from_ner(batch[j], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fill remaining with failed results
            while len(results) < len(addresses):
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"Processing error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        success_count = sum(1 for r in results if r.parse_success)
        
        print(f"✅ NVIDIA GPU completed {len(addresses)} addresses in {processing_time:.2f}s")
        print(f"   Success rate: {success_count}/{len(addresses)} ({success_count/len(addresses)*100:.1f}%)")
        print(f"   Speed: {len(addresses)/processing_time:.1f} addresses/second")
        
        return results
    
    def _clean_address_text(self, address: str) -> str:
        """Clean address text."""
        return re.sub(r'\s+', ' ', address.strip())[:300]
    
    def _extract_fields_from_ner(self, raw_address: str, entities) -> ParsedAddress:
        """Extract fields from NER entities."""
        # Handle both single entity and list of entities
        if not isinstance(entities, list):
            entities = [entities] if entities else []
        
        # Filter entities by confidence score
        filtered_entities = []
        for entity in entities:
            if isinstance(entity, dict) and entity.get('score', 0) > 0.5:
                filtered_entities.append(entity)
        
        fields = {
            'unit_number': '', 'society_name': '', 'landmark': '', 'road': '',
            'sub_locality': '', 'locality': '', 'city': '', 'district': '',
            'state': '', 'country': 'India', 'pin_code': ''
        }
        
        for entity in filtered_entities:
            entity_type = entity.get('entity_group', '').lower()
            entity_text = entity.get('word', '').strip().rstrip(',').strip()
            
            if entity_type in ['house_details', 'house_number', 'flat', 'unit'] and not fields['unit_number']:
                fields['unit_number'] = entity_text
            elif entity_type in ['building_name', 'building', 'society', 'complex'] and not fields['society_name']:
                fields['society_name'] = entity_text
            elif entity_type in ['landmarks', 'landmark', 'near'] and not fields['landmark']:
                fields['landmark'] = entity_text
            elif entity_type in ['street', 'road'] and not fields['road']:
                fields['road'] = entity_text
            elif entity_type in ['sublocality', 'sub_locality', 'area'] and not fields['sub_locality']:
                fields['sub_locality'] = entity_text
            elif entity_type in ['locality', 'neighbourhood', 'neighborhood'] and not fields['locality']:
                fields['locality'] = entity_text
            elif entity_type in ['city', 'town'] and not fields['city']:
                fields['city'] = entity_text
            elif entity_type in ['district'] and not fields['district']:
                fields['district'] = entity_text
            elif entity_type in ['state', 'province'] and not fields['state']:
                fields['state'] = entity_text
            elif entity_type in ['pincode', 'postcode', 'zip', 'pin_code'] and not fields['pin_code']:
                fields['pin_code'] = entity_text
        
        # Fallback PIN code extraction
        if not fields['pin_code']:
            pin_match = re.search(r'\b(\d{6})\b', raw_address)
            if pin_match:
                fields['pin_code'] = pin_match.group(1)
        
        if not fields['district'] and fields['city']:
            fields['district'] = fields['city']
        
        return ParsedAddress(
            unit_number=fields['unit_number'],
            society_name=fields['society_name'],
            landmark=fields['landmark'],
            road=fields['road'],
            sub_locality=fields['sub_locality'],
            locality=fields['locality'],
            city=fields['city'],
            district=fields['district'],
            state=fields['state'],
            country=fields['country'],
            pin_code=fields['pin_code'],
            note="Parsed using NVIDIA GPU Only",
            parse_success=False,
            parse_error=None
        )


def setup_logging():
    """Set up detailed logging for P1 processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'p1_nvidia_only_{timestamp}.log', encoding='utf-8')
        ]
    )


def load_p1_addresses(limit: int = 10000) -> List[str]:
    """Load addresses from export_customer_address_store_p1.csv."""
    
    print(f"📂 Loading addresses from export_customer_address_store_p1.csv (limit: {limit:,})...")
    
    try:
        # Load the P1 CSV file
        df = pd.read_csv('export_customer_address_store_p1.csv')
        
        print(f"✅ Loaded CSV with {len(df)} rows")
        
        # Check column names
        print(f"📋 Columns: {list(df.columns)}")
        
        # Find address column (common names)
        address_column = None
        for col in ['addr_text', 'address', 'full_address', 'Address', 'addr']:
            if col in df.columns:
                address_column = col
                break
        
        if not address_column:
            print("❌ Could not find address column. Available columns:")
            for col in df.columns:
                print(f"   - {col}")
            return []
        
        print(f"🎯 Using address column: '{address_column}'")
        
        # Filter out empty addresses
        df_clean = df.dropna(subset=[address_column])
        df_clean = df_clean[df_clean[address_column].str.strip() != '']
        
        # Limit to specified number
        if len(df_clean) > limit:
            df_clean = df_clean.head(limit)
            print(f"📊 Limited to first {limit:,} addresses")
        
        # Get all addresses
        addresses = df_clean[address_column].tolist()
        
        print(f"✅ Loaded {len(addresses):,} valid addresses for processing")
        return addresses
        
    except FileNotFoundError:
        print("❌ Error: export_customer_address_store_p1.csv not found")
        print("   Please ensure the file exists in the current directory")
        return []
    except Exception as e:
        print(f"❌ Error loading P1 file: {e}")
        return []


def process_p1_nvidia_only():
    """Process P1 file using NVIDIA GPU only."""
    
    print("🚀 PROCESSING P1 FILE WITH NVIDIA GPU ONLY")
    print("=" * 80)
    
    # System information
    print(f"🖥️  System Information:")
    print(f"   Configuration: NVIDIA GPU Only")
    print(f"   Address Limit: 10,000")
    print()
    
    # Load P1 addresses (limited to 10k)
    addresses = load_p1_addresses(limit=10000)
    
    if not addresses:
        print("❌ No addresses to process. Exiting.")
        return
    
    print(f"📊 Processing Details:")
    print(f"   Total Addresses: {len(addresses):,}")
    print(f"   GPU Configuration: Single NVIDIA GPU")
    print()
    
    # Configure NVIDIA parser
    print("🔧 Initializing NVIDIA GPU Parser...")
    
    parser = NvidiaOnlyParser(batch_size=16)
    
    if not parser.setup_nvidia_gpu():
        print("❌ Failed to initialize NVIDIA GPU. Exiting.")
        return None
    
    print(f"✅ Parser configured for NVIDIA GPU processing")
    print()
    
    # Start processing
    print("🚀 Starting P1 NVIDIA GPU Processing...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Process all addresses
        parsed_results = parser.parse_addresses(addresses)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_count = sum(1 for r in parsed_results if r.parse_success)
        failed_count = len(parsed_results) - success_count
        addresses_per_second = len(addresses) / total_time
        
        # Print comprehensive results
        print("\n🎉 P1 NVIDIA GPU PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"📊 PROCESSING RESULTS:")
        print(f"   Total Addresses: {len(addresses):,}")
        print(f"   Successfully Parsed: {success_count:,}")
        print(f"   Failed: {failed_count:,}")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        print()
        
        print(f"⏱️  TIMING RESULTS:")
        print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Speed: {addresses_per_second:.1f} addresses/second")
        print(f"   Speed: {addresses_per_second*60:.0f} addresses/minute")
        print()
        
        # Save results to CSV
        print("💾 SAVING RESULTS...")
        
        # Create results DataFrame
        results_data = []
        for i, (original_addr, parsed) in enumerate(zip(addresses, parsed_results)):
            results_data.append({
                'id': i + 1,
                'original_address': original_addr,
                'unit_number': parsed.unit_number if parsed.parse_success else '',
                'society_name': parsed.society_name if parsed.parse_success else '',
                'landmark': parsed.landmark if parsed.parse_success else '',
                'road': parsed.road if parsed.parse_success else '',
                'sub_locality': parsed.sub_locality if parsed.parse_success else '',
                'locality': parsed.locality if parsed.parse_success else '',
                'city': parsed.city if parsed.parse_success else '',
                'district': parsed.district if parsed.parse_success else '',
                'state': parsed.state if parsed.parse_success else '',
                'country': parsed.country if parsed.parse_success else '',
                'pin_code': parsed.pin_code if parsed.parse_success else '',
                'parse_success': parsed.parse_success,
                'parse_error': parsed.parse_error if not parsed.parse_success else '',
                'note': parsed.note if parsed.parse_success else ''
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'p1_nvidia_only_10k_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"✅ Results saved to: {output_file}")
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Configuration: NVIDIA GPU Only")
        print(f"   Addresses Processed: {len(addresses):,}")
        print(f"   Processing Speed: {addresses_per_second:.1f} addresses/second")
        print(f"   Total Time: {total_time/60:.1f} minutes")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        
        return {
            'total_addresses': len(addresses),
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': addresses_per_second,
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        logging.error(f"P1 NVIDIA GPU processing failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔧 P1 NVIDIA GPU Only Address Parser")
    print("Configuration: Single NVIDIA GPU, 10k addresses")
    print("=" * 80)
    
    try:
        results = process_p1_nvidia_only()
        
        if results:
            print(f"\n🎉 P1 Processing Complete!")
            print(f"Results saved to: {results['output_file']}")
        else:
            print(f"\n❌ P1 Processing failed. Check logs for details.")
        
        return results
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}", exc_info=True)
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()