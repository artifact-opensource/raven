import os
import requests
import json

SERVER_URL = "http://localhost:9000/infer"
TREASURY_PATH = "/home/adam/worxpace/av_treasury"

def audit_file(filename):
    with open(os.path.join(TREASURY_PATH, filename), "r") as f:
        content = f.read()
    
    prompt = f"Sovereign Audit for {filename}:\n\n{content}\n\nIdentify vulnerabilities and yield optimizations."
    payload = {"prompt": prompt}
    
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        return response.json().get("response", "No response")
    except Exception as e:
        return f"Error auditing {filename}: {str(e)}"

# Find all .sol files
sol_files = []
for root, dirs, files in os.walk(TREASURY_PATH):
    for file in files:
        if file.endswith(".sol"):
            sol_files.append(os.path.join(root, file))

# Run the audit
final_report = ""
for file_path in sol_files:
    print(f"Auditing {os.path.basename(file_path)}...")
    res = audit_file(file_path)
    final_report += f"### {os.path.basename(file_path)}\n{res}\n\n"

with open("/home/adam/worxpace/gladius/raven/docs/treasury_audit_report.md", "w") as f:
    f.write(final_report)

print("Audit Complete. Report saved to /home/adam/worxpace/gladius/raven/docs/treasury_audit_report.md")
