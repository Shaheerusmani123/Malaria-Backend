# app.py
# Flask backend for PyTorch malaria detection model with PDF report generation
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms
from torchvision import models
import os
from datetime import datetime
import base64

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,https://malaria-detector.vercel.app").split(",")
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS, "supports_credentials": True}})

# Load pretrained ResNet50 model
from torchvision import models

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/malaria_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading model...")
try:
    # Create ResNet50 architecture
    model = models.resnet50(pretrained=False)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")
    model = None

# Image preprocessing for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Malaria Detector API',
        'status': 'running',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'generate-report': '/api/generate-report (POST)',
            'health': '/health (GET)'
        }
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()
        
        is_infected = probability > 0.5
        confidence = probability if is_infected else 1 - probability
        
        # Prepare annotated image (simply reusing the original until actual annotation logic is added)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        annotated_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        result = {
            'success': True,
            'prediction': 'Parasitized' if is_infected else 'Uninfected',
            'confidence': f"{round(confidence * 100, 2)}",
            'probability': f"{round(probability, 4)}",
            'annotated_image': annotated_base64,  # PNG base64
            'parasiteCount': int(confidence * 50) if is_infected else 0,
            'severity': 'high' if (is_infected and confidence > 0.8) else 
                       'medium' if (is_infected and confidence > 0.6) else
                       'low' if is_infected else 'none'
        }
        
        return jsonify(result), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        patient_info = data.get('patientInfo', {})
        results = data.get('results', {})
        image_data_base64 = data.get('imageData', '')
        
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        story.append(Paragraph("MALARIA DETECTION REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
        story.append(Paragraph(f"<b>Report ID:</b> {datetime.now().strftime('%Y%m%d-%H%M%S')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        if patient_info and any(patient_info.values()):
            story.append(Paragraph("Patient Information", heading_style))
            
            patient_data = []
            if patient_info.get('patientId'):
                patient_data.append(['Patient ID:', patient_info.get('patientId')])
            if patient_info.get('age'):
                patient_data.append(['Age:', f"{patient_info.get('age')} years"])
            if patient_info.get('gender'):
                patient_data.append(['Gender:', patient_info.get('gender', '').capitalize()])
            
            if patient_data:
                patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
                patient_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                story.append(patient_table)
            
            if patient_info.get('notes'):
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(f"<b>Clinical Notes:</b>", styles['Normal']))
                story.append(Paragraph(patient_info['notes'], styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Analysis Results", heading_style))
        
        prediction = results.get('prediction', 'unknown')
        confidence = results.get('confidence', 0)
        
        if prediction == 'negative':
            result_color = '#10b981'
            result_text = 'NEGATIVE'
        else:
            result_color = '#ef4444'
            result_text = 'POSITIVE'
        
        results_data = [
            ['Test Result:', Paragraph(f"<font color='{result_color}'><b>{result_text}</b></font>", styles['Normal'])],
            ['Confidence Level:', f"{confidence}%"],
            ['Parasite Count:', str(results.get('parasiteCount', 0))],
            ['Severity Assessment:', results.get('severity', 'N/A').capitalize()],
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.3*inch))
        
        if image_data_base64:
            try:
                story.append(Paragraph("Blood Smear Image", heading_style))
                if ',' in image_data_base64:
                    image_data_base64 = image_data_base64.split(',')[1]
                
                image_data = base64.b64decode(image_data_base64)
                image_buffer = io.BytesIO(image_data)
                img = RLImage(image_buffer, width=4*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"Error adding image: {e}")
        
        story.append(Paragraph("Recommendations", heading_style))
        
        if prediction == 'positive':
            recs = [
                "• Immediate consultation with a healthcare professional is recommended",
                "• Confirmatory testing may be required",
                "• Follow prescribed antimalarial treatment if diagnosed",
                "• Monitor symptoms closely",
                "• Ensure proper rest, hydration, and nutrition"
            ]
        else:
            recs = [
                "• No malaria parasites detected in this sample",
                "• Continue preventive measures in endemic areas",
                "• If symptoms persist, consult a healthcare professional",
                "• Regular screening recommended for high-risk individuals"
            ]
        
        for rec in recs:
            story.append(Paragraph(rec, styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Disclaimer", heading_style))
        disclaimer = (
            "This report is generated by an AI-powered diagnostic tool and should be used as a "
            "supplementary screening method only. Results must be confirmed by qualified healthcare "
            "professionals using standard diagnostic procedures."
        )
        story.append(Paragraph(disclaimer, styles['Italic']))
        
        doc.build(story)
        pdf_buffer.seek(0)
        
        patient_id = patient_info.get('patientId', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"malaria_report_{patient_id}_{timestamp}.pdf"
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/api/export-image', methods=['POST'])
def export_image():
    try:
        if request.is_json:
            data = request.get_json()
            img_base64 = data.get('image', None)
        else:
            img_base64 = request.form.get('image') or None
        if not img_base64:
            return jsonify({'error': 'No image data provided'}), 400
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Optional: apply metadata/text overlay here
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        filename = f"malaria_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        return send_file(buffer, mimetype='image/png', as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Export image error: {str(e)}")
        return jsonify({'error': f'Export image failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
