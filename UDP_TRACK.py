# udp_ie_to_json.py
import json, requests

API_KEY = "nd0Q8KO3RXhgGpvXv8RkrfGl6n9kJ61CgcEf4z2d"
BASE = "https://api.open.fec.gov/v1/schedules/schedule_e/"

def stream_udp_ie(cycle=2024):
    params = {
        "committee_id": "C00799031",   # United Democracy Project
        "cycle": cycle,
        "is_notice": False,
        "per_page": 100,
        "sort": "expenditure_date",
        "api_key": API_KEY,
    }
    while True:
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for row in data.get("results", []):
            yield row
        last = data.get("pagination", {}).get("last_indexes")
        if not last:
            break
        params.update(last)

with open("udp_ie_2024.json", "w", encoding="utf-8") as f:
    json.dump(list(stream_udp_ie(2024)), f, ensure_ascii=False, indent=2)

print("✅ Saved raw UDP independent expenditures to udp_ie_2024.json")