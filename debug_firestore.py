from google.cloud import firestore
from google.oauth2 import service_account
import os

# Use the service account key
KEY_PATH = "/Users/neilkhandelwal/Downloads/svc-key.json"
PROJECT_ID = "pactrack-d63e9"

creds = service_account.Credentials.from_service_account_file(KEY_PATH)
db = firestore.Client(project=PROJECT_ID, credentials=creds)

print("🔍 Checking candidates...")
# Try to get Dave Min
doc_ref = db.collection("candidates").document("H4CA47085")
doc = doc_ref.get()

if doc.exists:
    print(f"✅ Found document: {doc.id}")
    data = doc.to_dict()
    print(f"Has office field: {'office' in data}")
    print(f"Office: {data.get('office')}")
    print(f"State: {data.get('state_code')}")
    print(f"District: {data.get('district')}")
    print(data)
else:
    print(f"❌ Document H4CA47085 not found!")
    # List first 5 docs to see what IDs look like
    print("First 5 docs in collection:")
    for d in db.collection("candidates").limit(5).stream():
        data = d.to_dict()
        print(f"- {d.id}: office={data.get('office')}, district={data.get('district')}")
