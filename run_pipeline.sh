#!/bin/bash
# PAC Tracking Pipeline Runner

# Activate virtual environment
source venv/bin/activate

echo "🚀 Step 1/3: Fetching data from FEC API..."
python3 PAC_TRACK.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to fetch data"
    exit 1
fi

echo ""
echo "🚀 Step 2/3: Processing and enriching data..."
python3 process_data.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to process data"
    exit 1
fi

echo ""
echo "🚀 Step 3/3: Uploading to Firestore..."
python3 scripts.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to upload to Firestore"
    exit 1
fi

echo ""
echo "✅ Pipeline complete!"
