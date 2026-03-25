import pandas as pd
import os
from pathlib import Path

# Define path relative to this file
# src/ is parent, so ../data2 is correct
BASE_DIR = Path(__file__).parent.parent
MAPPING_FILE = BASE_DIR / "data2" / "reference" / "UNIFIED_MAPPING_REVIEW.csv"

def load_dynamic_vocabulary():
    """
    Reads the Unified Mapping CSV and returns a list of valid cell entities.
    Handles splitting of comma-separated values in specific columns.
    """
    try:
        if not MAPPING_FILE.exists():
            print(f"Warning: Mapping file not found at {MAPPING_FILE}")
            return []

        df = pd.read_csv(MAPPING_FILE)
        
        # Columns to scan for cell type names
        cols_to_scan = [
            'CANONICAL_NAME', 
            'INTERACTOME_GRANULAR_TYPES', 
            'CATEGORICAL_TYPES', 
            'QUERY_TERM',
            'ORIGINAL_NAME',
            'SIGNATURE_NAMES'
        ]
        
        vocabulary = set()
        
        for col in cols_to_scan:
            if col in df.columns:
                # Get unique content first
                values = df[col].dropna().unique().astype(str)
                
                for val in values:
                    # Split by comma if present, as some columns are lists
                    if "," in val:
                        parts = [p.strip() for p in val.split(",")]
                        vocabulary.update([p for p in parts if p])
                    else:
                        vocabulary.add(val.strip())
        
        # Remove empty strings if any
        vocabulary = {v for v in vocabulary if v}
        
        return sorted(list(vocabulary))
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return []

# Load once at module level
VALID_CELL_TYPES = load_dynamic_vocabulary()

# Limit the display in prompt to avoid token overflow but ensure critical ones are there
# We'll take a reasonable subset or all if it fits. 
# For 500 types, it might be ~1-2k tokens, which is fine for modern LLMs.
# But let's verify length.
vocab_str = ", ".join(VALID_CELL_TYPES)

SYSTEM_PROMPT_REWRITER = f"""
You are an expert biological query translator for a Pancreatic Cancer RAG system.
Your job is to translate user questions into EXACT database terms to improve search retrieval.

VALID DATABASE ENTITIES (Vocabulary):
{vocab_str}

RULES:
1. Map colloquial terms to exact database keys.
   - Example: "CD4-positive, CCL5-expressing" -> "CD4+CCL5+ Tm"
   - Example: "Exhausted T cells" -> "CD8 T EXHAUSTED" or "CD8+TCF7+ Tex"
   - Example: "Obesity" -> "BMI_Slope", "Obese_vs_Normal", "Overweight_vs_Normal"
2. Map "Th1" to "CD4_T_TH1" or related canonical types.
3. If the user asks about an interaction, identify the involved cell types precisely.
4. Output ONLY the improved search query. Do NOT answer the question.
5. If the user query is already specific, output it as is or slightly refined for keyword matching.
"""
