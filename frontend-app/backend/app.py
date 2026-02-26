"""
backend/app.py — Flask entry point.
Registers API blueprint and serves frontend templates.
"""
import os
import logging
from flask import Flask, render_template
from api import api

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

app.register_blueprint(api)

AWS_REGION    = os.environ.get("AWS_REGION",    "ap-south-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")

@app.route("/")
def index():
    return render_template("index.html",
                           endpoint_name=ENDPOINT_NAME,
                           region=AWS_REGION)

@app.route("/batch")
def batch():
    return render_template("batch.html",
                           endpoint_name=ENDPOINT_NAME,
                           region=AWS_REGION)

if __name__ == "__main__":
    print(f"\n  Endpoint : {ENDPOINT_NAME or 'NOT SET'}")
    print(f"  Region   : {AWS_REGION}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
