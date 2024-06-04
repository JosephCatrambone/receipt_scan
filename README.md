## Receipt Scan

Just playing around with structured LLM generation and OCR.

![An image of assorted receipts, some handwritten, and the resulting structured JSON.](.github/receipt.jpg?raw=true)

### Setting Up (Dev):

```bash
# Clone the repo:
git clone https://github.com/JosephCatrambone/receipt_scan
# OR
git clone git@github.com:JosephCatrambone/receipt_scan.git

# Initialize your virtual environment

# Install the requirements:
pip install -r ./requirements.txt

# Install fastapi-cli
pip install fastapi-cli

# Define your OpenAI API keys:
export OPENAI_API_KEY="not-a-real-api-key"

# Serve:
fastapi dev

# Access the site at http://localhost:8000
```
