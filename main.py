import argparse
import os
import sys
from src.image_processor import MetallographicAnalyzer
from src.reporter import Reporter

def main():
    parser = argparse.ArgumentParser(description="Metallographic Microstructure Analysis Tool")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="output", help="Directory to save results")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale in microns per pixel")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum grain area in pixels to keep")
    parser.add_argument("--min-distance", type=int, default=15, help="Minimum distance between grain centers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)
        
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading {args.input}...")
    analyzer = MetallographicAnalyzer(
        scale_um_per_pixel=args.scale,
        min_grain_area_px=args.min_area,
        min_distance=args.min_distance
    )
    analyzer.load_image(args.input)
    
    print("Preprocessing (Denoise + CLAHE)...")
    analyzer.preprocess()
    
    print("Segmenting Phases (Otsu)...")
    analyzer.segment_phases()
    
    print("Separating Grains (Watershed)...")
    analyzer.separate_grains()
    
    print("Extracting Boundaries...")
    analyzer.extract_boundaries()
    
    print("Analyzing Results...")
    results, metrics = analyzer.get_analysis_results()
    
    print("Generating Report and Overlays...")
    reporter = Reporter(analyzer, args.output)
    reporter.save_overlays()
    reporter.generate_report(results, metrics)
    
    print(f"Analysis complete. Results saved to {args.output}/")

if __name__ == "__main__":
    main()
