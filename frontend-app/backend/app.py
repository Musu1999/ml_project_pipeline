"""
backend/app.py — Flask entry point (v3.1)
FIX: Renamed page route functions (page_index, page_batch, page_monitor, page_analytics)
     to avoid Flask endpoint name collision with api Blueprint functions.
     api.py Blueprint defines monitor(), index(), batch() etc — Flask requires
     unique function names across app + all registered blueprints.
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
def page_index():
    return render_template("index.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

@app.route("/batch")
def page_batch():
    return render_template("batch.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

@app.route("/monitor")
def page_monitor():
    return render_template("monitor.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

@app.route("/analytics")
def page_analytics():
    return render_template("analytics.html", endpoint_name=ENDPOINT_NAME, region=AWS_REGION)

if __name__ == "__main__":
    print(f"\n  Template : {TEMPLATE_DIR}")
    print(f"  Static   : {STATIC_DIR}")
    print(f"  Endpoint : {ENDPOINT_NAME or 'NOT SET'}")
    print(f"  Region   : {AWS_REGION}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
