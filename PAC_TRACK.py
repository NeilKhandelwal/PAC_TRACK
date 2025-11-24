# PAC_TRACK.py
import json
import requests
import os
import time
from pathlib import Path

# Configuration
API_KEY = os.getenv("FEC_API_KEY")
if not API_KEY:
    raise ValueError("FEC_API_KEY not found in environment variables")
BASE_URL_E = "https://api.open.fec.gov/v1/schedules/schedule_e/"
BASE_URL_B = "https://api.open.fec.gov/v1/schedules/schedule_b/"
CONFIG_PATH = Path("config/pacs.json")
RAW_DATA_DIR = Path("data/raw")

def load_pacs():
    if not CONFIG_PATH.exists():
        print(f"❌ Config file not found: {CONFIG_PATH}")
        return []
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def stream_pac_ie(committee_id, cycle=2024):
    """Stream Schedule E (Independent Expenditures)"""
    params = {
        "committee_id": committee_id,
        "cycle": cycle,
        "is_notice": False,
        "per_page": 100,
        "sort": "expenditure_date",
        "api_key": API_KEY,
    }
    yield from _stream_endpoint(BASE_URL_E, params)

def stream_pac_sb(committee_id, cycle=2024):
    """Stream Schedule B (Operating Expenditures)"""
    params = {
        "committee_id": committee_id,
        "cycle": cycle,
        "per_page": 100,
        "sort": "disbursement_date",
        "api_key": API_KEY,
    }
    yield from _stream_endpoint(BASE_URL_B, params)

def _stream_endpoint(url, params):
    page_count = 0
    while True:
        try:
            # Retry loop for 429
            while True:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 429:
                    print("   ⚠️ Rate limit hit. Sleeping for 5s...")
                    time.sleep(5)
                    continue
                r.raise_for_status()
                break
                
            data = r.json()
            
            results = data.get("results", [])
            if not results:
                break
                
            for row in results:
                yield row
            
            page_count += 1
            if page_count % 10 == 0:
                print(f"   ... fetched {page_count} pages")
                
            last = data.get("pagination", {}).get("last_indexes")
            if not last:
                break
            params.update(last)
            
            # Rate limiting
            time.sleep(0.2)
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pacs = load_pacs()
    print(f"ℹ️ Found {len(pacs)} PACs to track.")
    
    for pac in pacs:
        name = pac.get("name")
        cid = pac.get("id")
        
        # 1. Schedule E (IE)
        print(f"🚀 Fetching Schedule E (IE) for {name} ({cid})...")
        rows_e = list(stream_pac_ie(cid))
        out_path_e = RAW_DATA_DIR / f"{name}_ie.json"
        
        # Only write if we got data OR file doesn't exist (don't overwrite with empty)
        if rows_e or not out_path_e.exists():
            with open(out_path_e, "w", encoding="utf-8") as f:
                json.dump(rows_e, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved {len(rows_e)} IE records to {out_path_e}")
        else:
            print(f"⚠️ No data fetched for IE. Preserving existing file: {out_path_e}")

        # 2. Schedule B (Operating)
        print(f"🚀 Fetching Schedule B (Operating) for {name} ({cid})...")
        rows_b = list(stream_pac_sb(cid))
        out_path_b = RAW_DATA_DIR / f"{name}_sb.json"
        
        # Only write if we got data OR file doesn't exist
        if rows_b or not out_path_b.exists():
            with open(out_path_b, "w", encoding="utf-8") as f:
                json.dump(rows_b, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved {len(rows_b)} SB records to {out_path_b}")
        else:
            print(f"⚠️ No data fetched for SB. Preserving existing file: {out_path_b}")

if __name__ == "__main__":
    main()
