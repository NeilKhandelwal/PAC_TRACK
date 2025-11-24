from google.cloud import firestore
from google.oauth2 import service_account
import os

KEY_PATH = "/Users/neilkhandelwal/Downloads/svc-key.json"
PROJECT_ID = "pactrack-d63e9"

creds = service_account.Credentials.from_service_account_file(KEY_PATH)
db = firestore.Client(project=PROJECT_ID, credentials=creds)

doc = db.collection("meta").document("app").get()
if doc.exists:
    print(f"Meta Data: {doc.to_dict()}")
else:
    print("Meta doc does not exist")
