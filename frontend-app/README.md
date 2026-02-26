# AttritionIQ — HR Attrition Dashboard

Docker-based frontend for the HR Attrition ML Pipeline.

## Structure
```
frontend-app/
├── backend/
│   ├── api.py          # REST API Blueprint (/api/*)
│   └── app.py          # Flask entry point
├── frontend/
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/
│   │       ├── app.js      # Single predict logic
│   │       └── batch.js    # Batch predict logic
│   └── templates/
│       ├── index.html      # Single predict page
│       └── batch.html      # Batch predict page
├── Dockerfile.web
├── Dockerfile.api
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

1. Create `.env` file:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_SESSION_TOKEN=your_token
AWS_REGION=ap-south-1
ENDPOINT_NAME=mousumi-hr-attrition-endpoint
```

2. Run:
```bash
docker compose --env-file .env up -d --build
```

3. Open port 5000 in Codespace Ports tab.

## API Endpoints
- `GET  /api/health`         — Service health
- `GET  /api/model/info`     — Model metadata
- `GET  /api/features`       — Feature list
- `POST /api/predict`        — Single prediction
- `POST /api/predict/batch`  — Batch prediction (up to 50)
