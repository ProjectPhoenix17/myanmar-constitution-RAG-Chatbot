""""
backend_exact_match.py
Handles exact section retrieval and query parsing.
Strict numeric matching with sub-section fallback logic.
Now supports both Sections (ပုဒ်မ) and Tables (ဇယား).
"""

import json
import re
from typing import List, Dict

MM_TO_EN = {
    "၀": "0", "၁": "1", "၂": "2", "၃": "3", "၄": "4",
    "၅": "5", "၆": "6", "၇": "7", "၈": "8", "၉": "9"
}

def normalize(text: str) -> str:
    if not text:
        return ""
    for mm, en in MM_TO_EN.items():
        text = text.replace(mm, en)
    text = re.sub(r"\s+", " ", text)
    # Fix common typos and normalize parentheses to spaces for better matching
    text = text.replace("ပုတ္မ", "ပုဒ်မ").replace("ပုဒ္မ", "ပုဒ်မ")
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("（", " ").replace("）", " ")  # Full-width parentheses
    return text.strip()

def extract_base_number(text: str) -> int:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0

class ExactMatchEngine:
    CHAPTER_RANGES = {
        1: (1, 48), 2: (49, 56), 3: (57, 73), 4: (74, 198),
        5: (199, 292), 6: (293, 336), 7: (337, 344), 8: (345, 390),
        9: (391, 403), 10: (404, 409), 11: (410, 432), 12: (433, 436),
        13: (437, 440), 14: (441, 448), 15: (449, 457)
    }
    
    # 🔹 NEW: Define maximum limits
    MAX_SECTION = 457
    MAX_TABLE = 5
    MAX_CHAPTER = 15
    OUT_OF_RANGE_MESSAGE = "ဖွဲ့စည်းပုံအခြေခံဥပဒေ(၂၀၀၈)သည် အခန်း(၁၅)၊ပုဒ်မ ၄၅၇ ခုနဲ့ နောက်ဆက်တဲ့ဇယား ၅ခုထိသာ ပါရှိပါသည်။"

    def __init__(self, json_path: str):
        # Sections (ပုဒ်မ)
        self.exact_sections: Dict[str, Dict] = {}
        self.base_sections: Dict[int, List[Dict]] = {}
        self.max_section_num: int = 0
        
        # Tables (ဇယား)
        self.exact_tables: Dict[str, Dict] = {}
        self.base_tables: Dict[int, List[Dict]] = {}
        self.max_table_num: int = 0
        
        self._load_data(json_path)

    def _load_data(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            sec = item.get("Section", "").strip()
            ctx = item.get("Context", "").strip()
            table_field = item.get("Table", "").strip()  # New: Check for separate Table field
            
            # Determine if this is a table or section
            is_table = False
            identifier = ""
            
            # Check if Table field exists and is not empty
            if table_field:
                is_table = True
                identifier = table_field
            # Check if Section contains "ဇယား"
            elif sec and re.search(r"ဇယား", sec):
                is_table = True
                identifier = sec
            elif sec:
                identifier = sec
            
            if not identifier:
                continue
            
            norm_id = normalize(identifier)
            base_num = extract_base_number(norm_id)
            
            entry = {
                "section": identifier,  # Keep original name
                "context": ctx, 
                "number": base_num,
                "type": "table" if is_table else "section"  # Mark type
            }
            
            if is_table:
                # Store in tables dictionaries
                self.max_table_num = max(self.max_table_num, base_num)
                
                keys_to_store = {norm_id}
                clean_id = re.sub(r"^(?:ဇယား|table)\s*", "", norm_id, flags=re.IGNORECASE).strip()
                keys_to_store.add(clean_id)
                
                for key in keys_to_store:
                    self.exact_tables[key] = entry
                
                if base_num not in self.base_tables:
                    self.base_tables[base_num] = []
                self.base_tables[base_num].append(entry)
            else:
                # Store in sections dictionaries
                self.max_section_num = max(self.max_section_num, base_num)
                
                keys_to_store = {norm_id}
                clean_id = re.sub(r"^(?:ပုဒ်မ|section)\s*", "", norm_id, flags=re.IGNORECASE).strip()
                keys_to_store.add(clean_id)
                
                for key in keys_to_store:
                    self.exact_sections[key] = entry
                
                if base_num not in self.base_sections:
                    self.base_sections[base_num] = []
                self.base_sections[base_num].append(entry)

    def _parse_query_sections(self, query: str) -> List[str]:
        """Extract requested section/table identifiers from query."""
        q = normalize(query)
        requested_items = []
        
        # 🔹 1. Match sections with separators: "ပုဒ်မ 1/2/3", "ပုဒ်မ 1နဲ့2"
        pattern = r"((?:ပုဒ်မ|section)\s*[0-9]+(?:\s*(?:/|နဲ့|နှင့်|and|၊|,)\s*[0-9]+)*(?:\s*[a-zA-Zက-အ]+|\([^)]+\))?)"
        for m in re.finditer(pattern, q, re.IGNORECASE):
            sec_str = m.group(1).strip()
            
            if re.search(r"(?:/|နဲ့|နှင့်|and|၊|,)", sec_str):
                prefix_match = re.match(r"(ပုဒ်မ|section)\s*", sec_str, re.IGNORECASE)
                prefix = prefix_match.group(1) if prefix_match else "ပုဒ်မ"
                nums_part = sec_str[prefix_match.end():].strip()
                nums = re.split(r"\s*(?:/|နဲ့|နှင့်|and|၊|,)\s*", nums_part)
                for num in nums:
                    requested_items.append(f"{prefix} {num.strip()}")
            else:
                requested_items.append(sec_str)
        
        # 🔹 2. Match tables with separators: "ဇယား 1/2/3", "ဇယား 1နဲ့2"
        table_pattern = r"((?:ဇယား|table)\s*[0-9]+(?:\s*(?:/|နဲ့|နှင့်|and|၊|,)\s*[0-9]+)*(?:\s*[a-zA-Zက-အ]+|\([^)]+\))?)"
        for m in re.finditer(table_pattern, q, re.IGNORECASE):
            table_str = m.group(1).strip()
            
            if re.search(r"(?:/|နဲ့|နှင့်|and|၊|,)", table_str):
                prefix_match = re.match(r"(ဇယား|table)\s*", table_str, re.IGNORECASE)
                prefix = prefix_match.group(1) if prefix_match else "ဇယား"
                nums_part = table_str[prefix_match.end():].strip()
                nums = re.split(r"\s*(?:/|နဲ့|နှင့်|and|၊|,)\s*", nums_part)
                for num in nums:
                    requested_items.append(f"{prefix} {num.strip()}")
            else:
                requested_items.append(table_str)
            
        # 🔹 3. Handle ranges like "59 မှ 60 ထိ" (for sections)
        for m in re.finditer(r"([0-9]+)\s*မှ\s*([0-9]+)\s*ထိ?", q):
            start, end = int(m.group(1)), int(m.group(2))
            # Check context to determine if it's table or section
            context_before = q[:m.start()]
            if re.search(r"ဇယား|table", context_before, re.IGNORECASE):
                for i in range(start, end + 1):
                    requested_items.append(f"ဇယား {i}")
            else:
                for i in range(start, end + 1):
                    requested_items.append(f"ပုဒ်မ {i}")
                
        # 🔹 4. Handle ranges like "59-60" (for sections)
        for m in re.finditer(r"([0-9]+)\s*-\s*([0-9]+)", q):
            start, end = int(m.group(1)), int(m.group(2))
            context_before = q[:m.start()]
            if re.search(r"ဇယား|table", context_before, re.IGNORECASE):
                for i in range(start, end + 1):
                    requested_items.append(f"ဇယား {i}")
            else:
                for i in range(start, end + 1):
                    requested_items.append(f"ပုဒ်မ {i}")
                
        # 🔹 5. Handle chapters
        for m in re.finditer(r"(?:အခန်း|chapter)\s*([0-9]+)", q, re.IGNORECASE):
            ch = int(m.group(1))
            if ch in self.CHAPTER_RANGES:
                start, end = self.CHAPTER_RANGES[ch]
                for i in range(start, end + 1):
                    requested_items.append(f"ပုဒ်မ {i}")
                    
        return list(set(requested_items))

    def _validate_ranges(self, requested: List[str]) -> bool:
        """Validate if requested items are within defined ranges."""
        for req in requested:
            norm_req = normalize(req)
            base_num = extract_base_number(norm_req)
            
            # Check if it's a chapter request
            if re.search(r"အခန်း|chapter", norm_req, re.IGNORECASE):
                if base_num > self.MAX_CHAPTER:
                    return False
            # Check if it's a table request
            elif re.search(r"ဇယား|table", norm_req, re.IGNORECASE):
                if base_num > self.MAX_TABLE:
                    return False
            # Otherwise it's a section request
            else:
                if base_num > self.MAX_SECTION:
                    return False
        
        return True

    def search(self, query: str) -> Dict:
        intent_pattern = re.compile(r"(?:ပုဒ်မ|အခန်း|section|chapter|ပုဒ်|ဇယား|table)", re.IGNORECASE)
        if not intent_pattern.search(query):
            return {"error": "ပုဒ်မ၊ ဇယား သို့မဟုတ် အခန်းကို ရှာဖွေရန် ဖော်ပြပေးပါ။"}

        requested = self._parse_query_sections(query)
        
        # 🔹 NEW: Validate ranges before searching
        if not self._validate_ranges(requested):
            return {"error": self.OUT_OF_RANGE_MESSAGE}
        
        results = []
        missing = []

        for req in requested:
            norm_req = normalize(req)
            base_num = extract_base_number(norm_req)
            
            # Determine if this is a table or section request
            is_table_request = re.search(r"ဇယား|table", norm_req, re.IGNORECASE) is not None
            
            if is_table_request:
                # Search in tables
                clean_req = re.sub(r"^(?:ဇယား|table)\s*", "", norm_req, flags=re.IGNORECASE).strip()
                
                if norm_req in self.exact_tables:
                    results.append(self.exact_tables[norm_req])
                elif clean_req in self.exact_tables:
                    results.append(self.exact_tables[clean_req])
                else:
                    found_base = False
                    if base_num in self.base_tables:
                        for entry in self.base_tables[base_num]:
                            norm_sec = normalize(entry["section"])
                            clean_sec = re.sub(r"^(?:ဇယား|table)\s*", "", norm_sec, flags=re.IGNORECASE).strip()
                            
                            if clean_sec == str(base_num):
                                results.append(entry)
                                found_base = True
                                break
                    
                    if not found_base:
                        missing.append(req)
            else:
                # Search in sections
                clean_req = re.sub(r"^(?:ပုဒ်မ|section)\s*", "", norm_req, flags=re.IGNORECASE).strip()
                
                if norm_req in self.exact_sections:
                    results.append(self.exact_sections[norm_req])
                elif clean_req in self.exact_sections:
                    results.append(self.exact_sections[clean_req])
                else:
                    found_base = False
                    if base_num in self.base_sections:
                        for entry in self.base_sections[base_num]:
                            norm_sec = normalize(entry["section"])
                            clean_sec = re.sub(r"^(?:ပုဒ်မ|section)\s*", "", norm_sec, flags=re.IGNORECASE).strip()
                            
                            if clean_sec == str(base_num):
                                results.append(entry)
                                found_base = True
                                break
                    
                    if not found_base:
                        missing.append(req)

        # Deduplicate results by original section name
        seen = set() 
        unique_results = []
        for r in results:
            if r["section"] not in seen:
                seen.add(r["section"])
                unique_results.append(r)

        # Sort results: first by type (section/table), then by number, then by name
        unique_results.sort(key=lambda x: (
            0 if x.get("type") == "section" else 1,  # Sections first
            x.get("number", 0), 
            x.get("section", "")
        ))

        return {"results": unique_results, "missing": sorted(list(set(missing)))}