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
    # Serve the generated dashboard
    return send_from_directory(OUTPUT_FOLDER, 'index.html')

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
