import os
from flask import Flask, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from src.image_processor import MetallographicAnalyzer
from src.reporter import Reporter
import cv2

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'final_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app = Flask(__name__, static_folder=OUTPUT_FOLDER, static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Check if report exists
    report_path = os.path.join(OUTPUT_FOLDER, 'index.html')
    if os.path.exists(report_path):
        return send_from_directory(OUTPUT_FOLDER, 'index.html')
    
    # Fallback Landing Page
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Microstructure Analysis</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #f1f5f9; margin: 0; }
            .card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; width: 100%; max-width: 400px; }
            h1 { margin-bottom: 10px; color: #0f172a; font-size: 1.5rem; }
            p { color: #64748b; margin-bottom: 30px; }
            .upload-box { border: 2px dashed #cbd5e1; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            button { background: #0ea5e9; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 1rem; width: 100%; transition: background 0.2s; }
            button:hover { background: #0284c7; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Microstructure Analyzer</h1>
            <p>Automated Pearlite/Ferrite Segmentation</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="upload-box">
                    <input type="file" name="file" accept=".png,.jpg,.jpeg,.tif,.tiff" required>
                </div>
                <button type="submit">Analyze Image</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run Analysis
        try:
            print(f"Analyzing {filename}...")
            analyzer = MetallographicAnalyzer(
                scale_um_per_pixel=0.5, # Default scale, could be form input
                min_grain_area_px=100,
                min_distance=20
            )
            analyzer.load_image(filepath)
            analyzer.preprocess()
            analyzer.segment_phases()
            analyzer.separate_grains()
            analyzer.extract_boundaries()
            results, metrics = analyzer.get_analysis_results()
            
            # Generate Report (Updates index.html in final_output)
            reporter = Reporter(analyzer, OUTPUT_FOLDER)
            reporter.save_overlays()
            reporter.generate_report(results, metrics)
            
            return redirect('/')
            
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return redirect('/')

if __name__ == '__main__':
    print("Starting server at http://localhost:8000")
    # Run main.py once if index.html doesn't exist? 
    # User can run main.py manually or we assume it exists.
    app.run(port=8000, debug=True)
