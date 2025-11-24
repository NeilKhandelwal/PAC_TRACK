import json
import os
import requests
import time
import sys
from pathlib import Path

# Configuration
API_KEY = os.getenv("FEC_API_KEY")
if not API_KEY:
    print("⚠️  FEC_API_KEY not found in environment variables")
    sys.exit(1)
CONFIG_PATH = Path("config/pacs.json")

def load_config():
    if not CONFIG_PATH.exists():
        return []
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(pacs):
    with open(CONFIG_PATH, "w") as f:
        json.dump(pacs, f, indent=4)
    print(f"✅ Updated {CONFIG_PATH}")

def search_candidate(name):
    """Find FEC Candidate ID by name"""
    url = "https://api.open.fec.gov/v1/candidates/search/"
    params = {
        "api_key": API_KEY,
        "q": name,
        "sort": "name"
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            print(f"⚠️ No candidate found for '{name}'")
            return None
        
        # Simple heuristic: prefer federal offices, take first result
        # In a real app, might want to ask user to disambiguate
        best = results[0]
        print(f"ℹ️ Found candidate: {best['name']} ({best['candidate_id']})")
        return best['candidate_id']
    except Exception as e:
        print(f"❌ Error searching for candidate '{name}': {e}")
        return None

def get_committee_name(committee_id):
    """Fetch committee name by ID"""
    url = f"https://api.open.fec.gov/v1/committee/{committee_id}/"
    params = {"api_key": API_KEY}
    try:
        r = requests.get(url, params=params)
        if r.status_code == 429:
            time.sleep(2)
            r = requests.get(url, params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0].get("name")
    except Exception:
        pass
    return None

def find_pacs_for_candidate(candidate_id):
    """Find PACs (committees) that made Independent Expenditures for this candidate"""
    url = "https://api.open.fec.gov/v1/schedules/schedule_e/"
    params = {
        "api_key": API_KEY,
        "candidate_id": candidate_id,
        "cycle": 2024,
        "per_page": 100,
        "sort": "expenditure_amount",
        "sort_hide_null": True
    }
    
    pacs_found = {}
    
    print(f"🔎 Searching for PACs spending on {candidate_id}...")
    try:
        pages = 0
        while True:
            r = requests.get(url, params=params)
            
            if r.status_code == 429:
                print("   ⚠️ Rate limit hit. Sleeping for 5s...")
                time.sleep(5)
                continue
                
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            
            for i, row in enumerate(results):
                cmte_id = row.get("committee_id")
                cmte_name = row.get("committee_name")
                
                if cmte_id:
                    if cmte_id not in pacs_found:
                        # If name is missing, try to fetch it
                        if not cmte_name:
                            cmte_name = get_committee_name(cmte_id)
                        
                        if cmte_name:
                            pacs_found[cmte_id] = cmte_name
            
            last = data.get("pagination", {}).get("last_indexes")
            pages += 1
            if not last or len(pacs_found) > 20 or pages > 50: # Limit to avoid scraping forever
                break
            params.update(last)
            time.sleep(0.2) # Be polite
            
    except Exception as e:
        print(f"❌ Error fetching expenditures: {e}")
            
    except Exception as e:
        print(f"❌ Error fetching expenditures: {e}")
        
    return pacs_found

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 find_pacs.py \"Candidate Name 1\" \"Candidate Name 2\" ...")
        sys.exit(1)
        
    candidates = sys.argv[1:]
    existing_pacs = load_config()
    existing_ids = {p["id"] for p in existing_pacs}
    
    new_pacs_count = 0
    
    for name in candidates:
        cand_id = search_candidate(name)
        if not cand_id:
            continue
            
        found = find_pacs_for_candidate(cand_id)
        if not found:
            print(f"⚠️ No PAC spending found for {name} in 2024 cycle.")
            continue
            
        print(f"✅ Found {len(found)} PACs spending on {name}.")
        
        for pid, pname in found.items():
            if pid not in existing_ids:
                print(f"   + Adding new PAC: {pname} ({pid})")
                existing_pacs.append({
                    "name": pname.replace(" ", "_").upper()[:30], # Simple slug
                    "id": pid,
                    "description": pname
                })
                existing_ids.add(pid)
                new_pacs_count += 1
            else:
                # print(f"   . Skipping known PAC: {pname}")
                pass
                
    if new_pacs_count > 0:
        save_config(existing_pacs)
        print(f"🎉 Added {new_pacs_count} new PACs to tracking config.")
    else:
        print("ℹ️ No new PACs added (all found were already tracked).")

if __name__ == "__main__":
    main()
