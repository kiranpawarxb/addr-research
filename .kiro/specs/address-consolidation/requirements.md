# Requirements Document

## Introduction

This document specifies requirements for an Address Consolidation System that processes CSV files containing address data and uses LLM-based parsing to extract structured address components. The system parses unstructured Indian address text into standardized fields (Unit Number, Society Name, Landmark, Road, Sub-locality, Locality, City, District, State, Country, PIN code) and then consolidates addresses based on Society Name and other matching criteria.

## Glossary

- **Address Consolidation System**: The software system that processes address data and groups related addresses
- **Address Record**: A single row in the CSV file containing address information
- **Parsed Address**: A structured representation of an address with extracted components (UN, SN, LN, RD, SL, LOC, CY, DIS, ST, CN, PIN, Note)
- **UN (Unit Number)**: The flat, apartment, or unit identifier within a building
- **SN (Society Name)**: The name of the housing society, building, or residential complex
- **LN (Landmark)**: A nearby notable location or reference point
- **RD (Road)**: The street or road name
- **SL (Sub-locality)**: The sub-area or neighborhood within a locality
- **LOC (Locality)**: The broader area or locality name
- **CY (City)**: The city name
- **DIS (District)**: The district name
- **ST (State)**: The state name
- **CN (Country)**: The country name (typically "India")
- **PIN (PIN code)**: The postal index number
- **Note**: Additional information or parsing notes from the LLM
- **Consolidated Group**: A collection of Address Records that share the same Society Name and geographic identifiers
- **Address Text**: The unstructured string containing the full address information
- **LLM Parser**: The language model component that extracts structured fields from raw address text

## Requirements

### Requirement 1

**User Story:** As a data analyst, I want to load address data from a CSV file, so that I can process and consolidate addresses programmatically.

#### Acceptance Criteria

1. WHEN the user provides a CSV file path, THE Address Consolidation System SHALL read and parse the CSV file into Address Records
2. WHEN the CSV file contains required columns (addr_text, pincode, city_id), THE Address Consolidation System SHALL successfully load all records
3. IF the CSV file is missing required columns, THEN THE Address Consolidation System SHALL report an error with the missing column names
4. WHEN the CSV file contains malformed rows, THE Address Consolidation System SHALL skip invalid rows and log warnings with row numbers
5. WHEN loading completes, THE Address Consolidation System SHALL report the total count of successfully loaded Address Records

### Requirement 2

**User Story:** As a data analyst, I want to parse raw address text into structured components using an LLM, so that I can extract standardized address fields for consolidation.

#### Acceptance Criteria

1. WHEN an Address Record is processed, THE Address Consolidation System SHALL send the address text to the LLM Parser with the standardized prompt and examples
2. WHEN the LLM Parser returns a response, THE Address Consolidation System SHALL extract JSON fields (UN, SN, LN, RD, SL, LOC, CY, DIS, ST, CN, PIN, Note) into a Parsed Address
3. WHEN the LLM Parser fails to extract a field, THE Address Consolidation System SHALL store an empty string or "NA" for that field
4. WHEN parsing completes, THE Address Consolidation System SHALL validate that required fields (SN, CY, PIN) are present
5. WHEN the LLM Parser response is invalid JSON, THE Address Consolidation System SHALL log an error and mark the Address Record as unparseable

### Requirement 3

**User Story:** As a data analyst, I want to group addresses by their Society Name and geographic identifiers, so that I can see which addresses belong to the same physical location.

#### Acceptance Criteria

1. WHEN Parsed Addresses share the same Society Name (SN), THE Address Consolidation System SHALL group them into a single Consolidated Group
2. WHEN grouping Parsed Addresses, THE Address Consolidation System SHALL also match on PIN code to ensure geographic proximity
3. WHEN creating Consolidated Groups, THE Address Consolidation System SHALL preserve all original Address Record data and all Parsed Address fields
4. WHEN consolidation completes, THE Address Consolidation System SHALL report the total number of Consolidated Groups created
5. WHEN a Society Name cannot be extracted or is empty, THE Address Consolidation System SHALL place the Address Record in an "unmatched" group

### Requirement 4

**User Story:** As a data analyst, I want to export parsed address components to a CSV file, so that I can review the structured data and consolidation results.

#### Acceptance Criteria

1. WHEN the user requests export, THE Address Consolidation System SHALL write Parsed Addresses to an output CSV file
2. WHEN writing output, THE Address Consolidation System SHALL include all structured fields (UN, SN, LN, RD, SL, LOC, CY, DIS, ST, CN, PIN, Note) as separate columns
3. WHEN writing output, THE Address Consolidation System SHALL include a group identifier column for each Consolidated Group
4. WHEN writing output, THE Address Consolidation System SHALL preserve all original columns from the input CSV
5. WHEN export completes, THE Address Consolidation System SHALL report the output file path and record count

### Requirement 5

**User Story:** As a data analyst, I want to see statistics about the consolidation results, so that I can understand the quality and distribution of the groupings.

#### Acceptance Criteria

1. WHEN consolidation completes, THE Address Consolidation System SHALL calculate the total number of Consolidated Groups
2. WHEN consolidation completes, THE Address Consolidation System SHALL calculate the average number of Address Records per Consolidated Group
3. WHEN consolidation completes, THE Address Consolidation System SHALL identify the largest Consolidated Group by record count
4. WHEN consolidation completes, THE Address Consolidation System SHALL calculate the percentage of Address Records that were successfully matched
5. WHEN statistics are generated, THE Address Consolidation System SHALL display them in a readable format

### Requirement 6

**User Story:** As a data analyst, I want to configure the LLM parser and matching rules, so that I can fine-tune the parsing and consolidation behavior for my specific dataset.

#### Acceptance Criteria

1. WHERE an LLM API endpoint is configured, THE Address Consolidation System SHALL use that endpoint for parsing address text
2. WHERE fuzzy matching is enabled, THE Address Consolidation System SHALL use string similarity algorithms to match Society Names
3. WHERE a similarity threshold is configured, THE Address Consolidation System SHALL only group Society Names that exceed the threshold
4. WHERE batch processing is enabled, THE Address Consolidation System SHALL process multiple addresses in parallel to improve performance
5. WHEN configuration is invalid, THE Address Consolidation System SHALL report errors and use default values

### Requirement 7

**User Story:** As a data analyst, I want to handle parsing errors gracefully, so that the system can continue processing even when some addresses fail to parse.

#### Acceptance Criteria

1. WHEN the LLM Parser times out, THE Address Consolidation System SHALL retry the request up to three times
2. WHEN the LLM Parser returns an error after retries, THE Address Consolidation System SHALL log the error and mark the address as unparseable
3. WHEN an address is marked as unparseable, THE Address Consolidation System SHALL include it in the output with empty parsed fields
4. WHEN parsing errors occur, THE Address Consolidation System SHALL continue processing remaining addresses
5. WHEN processing completes, THE Address Consolidation System SHALL report the count of successfully parsed and failed addresses
