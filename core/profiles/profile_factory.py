import json
from pathlib import Path
from typing import Dict, Any

PROFILE_DIR = Path(__file__).resolve().parent

class ProfileFactory:
    @staticmethod
    def get_profile(person_type: str) -> Dict[str, Any]:
        if not person_type:
            person_type = "DEMENTIA"
            
        file_path = PROFILE_DIR / f"{person_type.lower()}_profile.json"
        
        if not file_path.exists():
            # Fallback to default (DEMENTIA)
            file_path = PROFILE_DIR / "dementia_profile.json"
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
