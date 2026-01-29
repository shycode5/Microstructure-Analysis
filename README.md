# Metallographic Analysis Dashboard

A computer vision application for automated microstructure analysis (Pearlite/Ferrite segmentation), grain quantification, and reporting.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Microstructure+Analysis+Dashboard "Run locally to see the dashboard")

## Features
- **Automated Phase Segmentation**: Uses Otsu's thresholding + Watershed algorithm to separate Pearlite (dark) and Ferrite (light) phases.
- **ML Correction Lens**: Lightweight Random Forest classifier refines ambiguous boundaries on-the-fly.
- **Grain Statistics**: Calculates grain count, area fractions, and mean grain size (ASTM E112 style metrics).
- **Interactive Dashboard**: Web interface for uploading images and viewing results (Segmentation, Boundaries, Stats).
- **Docker Ready**: Includes `Dockerfile` for easy deployment.

## Quick Start (Local)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**
   ```bash
   python server.py
   ```

3. **Open Dashboard**
   Go to [http://localhost:8000](http://localhost:8000)

## Deployment

### Render / Railway / Fly.io (Docker)
This project includes a `Dockerfile` compatible with most container-based hosting platforms.

1. Push this repo to GitHub.
2. Connect your repository to Render/Railway.
3. The platform will automatically detect the Dockerfile and build the image.

**Note**: The app requires `opencv-python-headless` which is included in `requirements.txt`.

## Project Structure
- `src/image_processor.py`: Core CV logic (Otsu, Watershed, ML Refinement).
- `src/reporter.py`: Generates HTML dashboard and visual overlays.
- `server.py`: Flask web server entry point.
- `main.py`: CLI entry point for batch processing.
