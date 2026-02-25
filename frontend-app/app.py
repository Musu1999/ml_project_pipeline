"""
app.py — HR Attrition Dashboard entry point
Serves the frontend and registers the API blueprint from api.py
"""
import os
import logging
from flask import Flask, render_template
from api import api   # import blueprint from api.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

# ── Register API blueprint (/api/...) ─────────────────────────────────────────
app.register_blueprint(api)

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION    = os.environ.get("AWS_REGION",    "ap-south-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")

# ── Frontend route ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           endpoint_name=ENDPOINT_NAME,
                           region=AWS_REGION)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  AttritionIQ — HR Attrition Dashboard")
    print("  Mousumi · SageMaker · Docker")
    print("═"*55)
    print(f"  Endpoint : {ENDPOINT_NAME or '⚠ NOT SET'}")
    print(f"  Region   : {AWS_REGION}")
    print(f"  API      : http://localhost:5000/api/health")
    print(f"  Docs     : http://localhost:5000/api/features")
    print("═"*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
