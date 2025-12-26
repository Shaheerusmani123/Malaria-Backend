# Malaria Detector Backend

## Installation & Setup

1. Install Python requirements
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Configure environment variables (see `.env`):
   ```env
   MODEL_PATH=./models/malaria_model.pth
   CORS_ORIGINS=http://localhost:5173,https://malaria-detector.vercel.app
   ```

## Running Locally
```
python app.py
# or
flask run
```

## API Endpoints

### POST `/api/predict`
- Form-data: `file` (image)
- Returns:
  `{ success, prediction: "Parasitized"|"Uninfected", confidence, probability, annotated_image (base64, optional) }`

### POST `/api/generate-report`
- JSON body: `{ patientInfo, results, imageData }`
- Returns: PDF (Content-Disposition: attachment)

### POST `/api/export-image`
- Receives: base64 or binary of annotated image
- Returns: PNG/JPEG (Content-Disposition: attachment)

## Deployment (Render)
- Uses `render.yaml`
- Set environment variables in Render dashboard
- Start command: `gunicorn app:app`
- Ensure model file is uploaded to `/models` directory

## Notes
- CORS only allows Vercel frontend and localhost
- Model loads on boot; fast inference
- Detailed errors and logging for network/API issues
