"""
backend/app.py — Flask entry point (Enhanced v3.0)
Added: /monitor route
"""
import os, logging
from flask import Flask, render_template
from api import api

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'frontend', 'templates')
STATIC_DIR   = os.path.join(BASE_DIR, '..', 'frontend', 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.register_blueprint(api)

AWS_REGION    = os.environ.get("AWS_REGION",    "ap-south-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")

@app.route("/")
def index():
    return render_template("index.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

@app.route("/batch")
def batch():
    return render_template("batch.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

@app.route("/monitor")
def monitor():
    return render_template("monitor.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

if __name__ == "__main__":
    print(f"\n  Template : {TEMPLATE_DIR}")
    print(f"  Static   : {STATIC_DIR}")
    print(f"  Endpoint : {ENDPOINT_NAME or 'NOT SET'}")
    print(f"  Region   : {AWS_REGION}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
