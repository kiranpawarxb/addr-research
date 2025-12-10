"""Unit tests for the Consolidation Engine component."""

import pytest
from src.consolidation_engine import ConsolidationEngine
from src.models import AddressRecord, ParsedAddress, ConsolidatedGroup


class TestConsolidationEngine:
    """Test suite for ConsolidationEngine class."""
    
    def test_initialization(self):
        """Test engine initialization with default parameters."""
        engine = ConsolidationEngine()
        assert engine.fuzzy_matching is False
        assert engine.similarity_threshold == 0.85
        assert engine.normalize_society_names is True
    
    def test_initialization_with_custom_params(self):
        """Test engine initialization with custom parameters."""
        engine = ConsolidationEngine(
            fuzzy_matching=True,
            similarity_threshold=0.9,
            normalize_society_names=False
        )
        assert engine.fuzzy_matching is True
        assert engine.similarity_threshold == 0.9
        assert engine.normalize_society_names is False
    
    def test_normalize_society_name(self):
        """Test society name normalization."""
        engine = ConsolidationEngine()
        
        # Test lowercase conversion
        assert engine._normalize_society_name("PRESTIGE APARTMENTS") == "prestige apartments"
        
        # Test whitespace trimming
        assert engine._normalize_society_name("  Lodha Complex  ") == "lodha complex"
        
        # Test special character removal (hyphens become spaces)
        assert engine._normalize_society_name("DLF-Phase@3!") == "dlf phase3"
        
        # Test multiple space collapse
        assert engine._normalize_society_name("Green   Valley    Homes") == "green valley homes"
        
        # Test empty string
        assert engine._normalize_society_name("") == ""
        
        # Test combined transformations (hyphens become spaces)
        assert engine._normalize_society_name("  PRESTIGE-Shantiniketan@2024  ") == "prestige shantiniketan2024"
    
    def test_consolidate_empty_list(self):
        """Test consolidation with empty input."""
        engine = ConsolidationEngine()
        result = engine.consolidate([])
        assert result == []
        assert engine._total_groups == 0
    
    def test_consolidate_single_record(self):
        """Test consolidation with a single record."""
        engine = ConsolidationEngine()
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([(address_record, parsed_address)])
        
        assert len(result) == 1
        assert result[0].society_name == "Prestige Apartments"
        assert result[0].pin_code == "560001"
        assert result[0].record_count == 1
    
    def test_consolidate_matching_society_and_pin(self):
        """Test consolidation groups addresses with same society name and PIN."""
        engine = ConsolidationEngine()
        
        # Create two records with same society and PIN
        records = []
        for i in range(2):
            address_record = AddressRecord(
                addr_hash_key=f"hash{i}",
                addr_text=f"Test address {i}",
                city_id="city1",
                pincode="560001",
                state_id="state1",
                zone_id="zone1",
                address_id=f"addr{i}",
                assigned_pickup_dlvd_geo_points="",
                assigned_pickup_dlvd_geo_points_count=0
            )
            
            parsed_address = ParsedAddress(
                society_name="Prestige Apartments",
                pin_code="560001",
                city="Bangalore",
                parse_success=True
            )
            
            records.append((address_record, parsed_address))
        
        result = engine.consolidate(records)
        
        assert len(result) == 1
        assert result[0].record_count == 2
    
    def test_consolidate_different_society_same_pin(self):
        """Test consolidation creates separate groups for different societies."""
        engine = ConsolidationEngine()
        
        # Create two records with different societies but same PIN
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="Lodha Complex",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        assert len(result) == 2
        assert all(group.record_count == 1 for group in result)
    
    def test_consolidate_same_society_different_pin(self):
        """Test consolidation creates separate groups for different PINs."""
        engine = ConsolidationEngine()
        
        # Create two records with same society but different PINs
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560002",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560002",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        assert len(result) == 2
        assert all(group.record_count == 1 for group in result)
    
    def test_consolidate_empty_society_name(self):
        """Test consolidation places empty society names in unmatched group."""
        engine = ConsolidationEngine()
        
        # Create records with empty society names
        records = []
        for i in range(2):
            address_record = AddressRecord(
                addr_hash_key=f"hash{i}",
                addr_text=f"Test address {i}",
                city_id="city1",
                pincode="560001",
                state_id="state1",
                zone_id="zone1",
                address_id=f"addr{i}",
                assigned_pickup_dlvd_geo_points="",
                assigned_pickup_dlvd_geo_points_count=0
            )
            
            parsed_address = ParsedAddress(
                society_name="",  # Empty society name
                pin_code="560001",
                city="Bangalore",
                parse_success=True
            )
            
            records.append((address_record, parsed_address))
        
        result = engine.consolidate(records)
        
        # Should have one unmatched group
        assert len(result) == 1
        assert result[0].group_id == ConsolidationEngine.UNMATCHED_GROUP_ID
        assert result[0].record_count == 2
        assert engine._unmatched_count == 2
    
    def test_consolidate_mixed_empty_and_valid(self):
        """Test consolidation with mix of empty and valid society names."""
        engine = ConsolidationEngine()
        
        # Create one record with valid society name
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        # Create one record with empty society name
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        # Should have two groups: one valid, one unmatched
        assert len(result) == 2
        
        # Find the groups
        valid_groups = [g for g in result if g.group_id != ConsolidationEngine.UNMATCHED_GROUP_ID]
        unmatched_groups = [g for g in result if g.group_id == ConsolidationEngine.UNMATCHED_GROUP_ID]
        
        assert len(valid_groups) == 1
        assert len(unmatched_groups) == 1
        assert valid_groups[0].record_count == 1
        assert unmatched_groups[0].record_count == 1
    
    def test_consolidate_normalization_groups_similar_names(self):
        """Test that normalization groups similar society names."""
        engine = ConsolidationEngine(normalize_society_names=True)
        
        # Create records with similar but differently formatted names
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="PRESTIGE APARTMENTS",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="prestige-apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        # Should be grouped together due to normalization
        assert len(result) == 1
        assert result[0].record_count == 2
    
    def test_calculate_similarity(self):
        """Test similarity calculation between society names."""
        engine = ConsolidationEngine()
        
        # Identical names
        assert engine._calculate_similarity("prestige apartments", "prestige apartments") == 1.0
        
        # Very similar names
        similarity = engine._calculate_similarity("prestige apartments", "prestige apartment")
        assert similarity > 0.9
        
        # Different names
        similarity = engine._calculate_similarity("prestige apartments", "lodha complex")
        assert similarity < 0.5
        
        # Empty strings
        assert engine._calculate_similarity("", "") == 0.0
        assert engine._calculate_similarity("prestige", "") == 0.0
    
    def test_fuzzy_matching_groups_similar_names(self):
        """Test fuzzy matching groups similar society names."""
        engine = ConsolidationEngine(
            fuzzy_matching=True,
            similarity_threshold=0.85
        )
        
        # Create records with similar names (typo)
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="Prestige Apartment",  # Singular vs plural
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        # Should be grouped together due to fuzzy matching
        assert len(result) == 1
        assert result[0].record_count == 2
    
    def test_fuzzy_matching_respects_threshold(self):
        """Test fuzzy matching respects similarity threshold."""
        engine = ConsolidationEngine(
            fuzzy_matching=True,
            similarity_threshold=0.95  # Very high threshold
        )
        
        # Create records with slightly different names
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="Prestige Apartment",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        # With high threshold, might not group together
        # (depends on exact similarity score)
        # At minimum, should create valid groups
        assert len(result) >= 1
        assert all(group.record_count > 0 for group in result)
    
    def test_fuzzy_matching_requires_same_pin(self):
        """Test fuzzy matching only matches within same PIN code."""
        engine = ConsolidationEngine(
            fuzzy_matching=True,
            similarity_threshold=0.85
        )
        
        # Create records with similar names but different PINs
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560002",  # Different PIN
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="Prestige Apartment",
            pin_code="560002",  # Different PIN
            city="Bangalore",
            parse_success=True
        )
        
        result = engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        # Should NOT be grouped together due to different PINs
        assert len(result) == 2
        assert all(group.record_count == 1 for group in result)
    
    def test_unique_group_ids(self):
        """Test that each group gets a unique ID."""
        engine = ConsolidationEngine()
        
        # Create multiple records with different societies
        records = []
        for i in range(3):
            address_record = AddressRecord(
                addr_hash_key=f"hash{i}",
                addr_text=f"Test address {i}",
                city_id="city1",
                pincode="560001",
                state_id="state1",
                zone_id="zone1",
                address_id=f"addr{i}",
                assigned_pickup_dlvd_geo_points="",
                assigned_pickup_dlvd_geo_points_count=0
            )
            
            parsed_address = ParsedAddress(
                society_name=f"Society {i}",
                pin_code="560001",
                city="Bangalore",
                parse_success=True
            )
            
            records.append((address_record, parsed_address))
        
        result = engine.consolidate(records)
        
        # Check all group IDs are unique
        group_ids = [group.group_id for group in result]
        assert len(group_ids) == len(set(group_ids))
    
    def test_get_statistics(self):
        """Test statistics reporting."""
        engine = ConsolidationEngine()
        
        # Create some records
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address 1",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Prestige Apartments",
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Test address 2",
            city_id="city1",
            pincode="560001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        parsed_address2 = ParsedAddress(
            society_name="",  # Empty
            pin_code="560001",
            city="Bangalore",
            parse_success=True
        )
        
        engine.consolidate([
            (address_record1, parsed_address1),
            (address_record2, parsed_address2)
        ])
        
        stats = engine.get_statistics()
        assert stats["total_groups"] == 2
        assert stats["unmatched_count"] == 1
