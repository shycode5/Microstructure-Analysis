import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Reporter:
    def __init__(self, analyzer, output_dir):
        self.analyzer = analyzer
        self.output_dir = output_dir
        
    def save_overlays(self):
        if self.analyzer.original_image is None:
            return

        original = self.analyzer.original_image.copy()
        h, w = original.shape[:2]
        
        # Create colored overlay
        # Pearlite (Dark) -> Red/Blue (let's use Red)
        # Ferrite (Light) -> Green/Yellow (let's use Green)
        
        overlay = np.zeros_like(original)
        
        # Blue channel for Pearlite (BGR: Blue is channel 0) - "Blue/Red" requested
        # Let's do Pearlite = Blue (255, 0, 0), Ferrite = Yellow (0, 255, 255)
        
        # BGR format
        pearlite_color = [0, 0, 255] # Red
        ferrite_color = [0, 255, 255] # Yellow
        
        # Mask boolean
        p_mask = self.analyzer.pearlite_mask > 0
        f_mask = self.analyzer.ferrite_mask > 0
        
        overlay[p_mask] = pearlite_color
        overlay[f_mask] = ferrite_color
        
        # Blend with original
        alpha = 0.4
        blended = cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0)
        
        # Draw boundaries
        # Boundaries are white (255) in self.analyzer.boundaries
        if self.analyzer.boundaries is not None:
            b_mask = self.analyzer.boundaries > 0
            blended[b_mask] = [0, 255, 0]
            
        cv2.imwrite(f"{self.output_dir}/original.png", original)
        cv2.imwrite(f"{self.output_dir}/segmented_color_overlay.png", blended)
        cv2.imwrite(f"{self.output_dir}/boundaries_overlay.png", self.analyzer.boundaries)

    def generate_report(self, results, metrics=None):
        df = pd.DataFrame(results)
        df.to_csv(f"{self.output_dir}/grains.csv", index=False)
        
        # Summary Stats
        total_area = df['area_um2'].sum()
        pearlite_stats = df[df['phase'] == 'Pearlite']
        ferrite_stats = df[df['phase'] == 'Ferrite']
        
        p_area = pearlite_stats['area_um2'].sum()
        f_area = ferrite_stats['area_um2'].sum()
        
        p_frac = (p_area / total_area * 100) if total_area > 0 else 0
        f_frac = (f_area / total_area * 100) if total_area > 0 else 0
        
        # Unpack metrics
        if metrics is None:
            metrics = {'otsu_confidence': 0, 'noise_rejection_percent': 0, 'over_segmentation_index': 0}
        
        otsu_conf = metrics.get('otsu_confidence', 0)
        noise_pct = metrics.get('noise_rejection_percent', 0)
        over_seg = metrics.get('over_segmentation_index', 0)

        # Markdown Report
        report_text = f"""# Metallographic Analysis Report

## Summary
- **Total Area Analyzed**: {total_area:.2f} um^2
- **Pearlite Area Fraction**: {p_frac:.2f}%
- **Ferrite Area Fraction**: {f_frac:.2f}%
- **Total Grains Detected**: {len(df)}

## Quality Metrics
- **Threshold Confidence (Otsu)**: {otsu_conf:.3f}
- **Noise Rejection**: {noise_pct:.1f}%
- **Over-segmentation Index**: {over_seg:.2f} (small/total ratio)

## Grain Statistics
| Phase | Count | Mean Area (um^2) |
|---|---|---|
| Pearlite | {len(pearlite_stats)} | {pearlite_stats['area_um2'].mean():.2f} |
| Ferrite | {len(ferrite_stats)} | {ferrite_stats['area_um2'].mean():.2f} |

See `grains.csv` for detailed per-grain data.
"""
        with open(f"{self.output_dir}/report.md", "w") as f:
            f.write(report_text)
            
        # HTML Dashboard (Compact Light Theme + Premium Metrics)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metallographic Analysis | AI Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --accent-primary: #0ea5e9;
            --accent-secondary: #6366f1;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --border: #e2e8f0;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 0;
            line-height: 1.4;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
            padding: 15px 25px;
        }}
        
        header {{
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 2px;
        }}
        
        /* Compact Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 16px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, var(--accent-primary), var(--accent-secondary));
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
        }}
        
        .stat-value {{
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 2px;
            color: var(--text-primary);
        }}
        
        .stat-unit {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 400;
        }}
        
        /* Confidence Panel */
        .quality-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }}

        /* Compact Images Grid */
        .images-section {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .image-card {{
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        
        .image-header {{
            padding: 10px 15px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .image-header h3 {{
            margin: 0;
            font-size: 0.95rem;
            font-weight: 600;
        }}
        
        .legend {{
            display: flex;
            gap: 8px;
            font-size: 0.75rem;
        }}
        
        .dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 4px;
        }}
        
        .img-container {{
            position: relative;
            width: 100%;
            height: 250px; 
            background: #f8fafc;
            overflow: hidden;
        }}
        
        .img-container img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        /* Compact Table Section */
        .details-section {{
            background: var(--card-bg);
            border-radius: 10px;
            border: 1px solid var(--border);
            padding: 15px;
            display: flex;
            gap: 20px;
            align-items: center;
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 0;
            min-width: 200px;
        }}
        
        .section-header h2 {{
            margin: 0;
            font-size: 1.1rem;
        }}
        
        .download-btn {{
            background: var(--text-primary);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.8rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th, td {{
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            color: var(--text-secondary);
            font-weight: 500;
            background: #f8fafc;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .phase-badge {{
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-pearlite {{ background: #fecaca; color: #b91c1c; }}
        .badge-ferrite {{ background: #fef08a; color: #854d0e; }}
        
        .header-upload-btn {
            background: white;
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .analyze-btn {
            background: var(--accent-primary);
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* Mobile Responsiveness */
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr 1fr;
            }}
            
            .quality-grid {{
                grid-template-columns: 1fr;
            }}
            
            .images-section {{
                grid-template-columns: 1fr;
            }}
            
            header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 15px;
            }}
            
            header form {{
                width: 100%;
                box-sizing: border-box;
            }}
            
            .details-section {{
                flex-direction: column;
                align-items: stretch;
                padding: 10px;
            }}
            
            .section-header {{
                justify-content: space-between;
                margin-bottom: 15px;
            }}
            
            .container {{
                padding: 10px 15px;
            }}
        }}
        
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Microstructure Analysis</h1>
                <div class="subtitle">Automated Phase Segmentation & Grain Quantification</div>
            </div>
            
            <!-- Upload Form -->
            <form action="/upload" method="post" enctype="multipart/form-data" style="display: flex; gap: 10px; align-items: center;">
                <input type="file" id="headerFile" name="file" accept="image/*,.tif,.tiff" required style="display: none;">
                <label for="headerFile" class="header-upload-btn">Choose File</label>
                <div id="headerFileName" style="font-size: 0.75rem; color: var(--text-secondary); max-width: 100px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: none;"></div>
                <button type="submit" class="analyze-btn">Analyze</button>
            </form>

            <script>
                document.getElementById('headerFile').addEventListener('change', function(e) {
                    if (e.target.files[0]) {
                        var name = e.target.files[0].name;
                        var label = document.getElementById('headerFileName');
                        label.textContent = name;
                        label.style.display = 'block';
                        document.querySelector('.header-upload-btn').style.background = '#e2e8f0';
                        document.querySelector('.header-upload-btn').style.color = '#0f172a';
                    }
                });
            </script>



        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Pearlite Fraction</div>
                <div class="stat-value" style="color: var(--danger)">{p_frac:.1f}<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Ferrite Fraction</div>
                <div class="stat-value" style="color: var(--warning)">{f_frac:.1f}<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Grains</div>
                <div class="stat-value">{len(df):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Analyzed Area</div>
                <div class="stat-value">{total_area/1000:.1f}<span class="stat-unit">k µm²</span></div>
            </div>
        </div>

        <!-- Advanced Quality Metrics -->
        <div class="quality-grid">
            <div class="stat-card" style="border-left: 4px solid var(--accent-secondary);">
                <div class="stat-label">Threshold Confidence</div>
                <div class="stat-value" style="font-size: 1.4rem;">{otsu_conf:.2f}<span class="stat-unit">/1.0</span></div>
                <div class="stat-unit">Histogram Sep.</div>
            </div>
            <div class="stat-card" style="border-left: 4px solid var(--success);">
                <div class="stat-label">Noise Rejected</div>
                <div class="stat-value" style="font-size: 1.4rem;">{noise_pct:.1f}<span class="stat-unit">%</span></div>
                <div class="stat-unit">&lt; {self.analyzer.min_area}px</div>
            </div>
            <div class="stat-card" style="border-left: 4px solid var(--warning);">
                <div class="stat-label">Over-seg. Index</div>
                <div class="stat-value" style="font-size: 1.4rem;">{over_seg:.2f}</div>
                <div class="stat-unit">Small/Total Ratio</div>
            </div>
        </div>

        <div class="images-section">
            <div class="image-card">
                <div class="image-header">
                    <h3>Original Specimen</h3>
                </div>
                <div class="img-container">
                    <img src="original.png" alt="Original">
                </div>
            </div>
            
            <div class="image-card">
                <div class="image-header">
                    <h3>Phase Segmentation</h3>
                    <div class="legend">
                        <div class="legend-item"><span class="dot" style="background: var(--danger)"></span> Pearlite</div>
                        <div class="legend-item"><span class="dot" style="background: var(--warning)"></span> Ferrite</div>
                    </div>
                </div>
                <div class="img-container">
                    <img src="segmented_color_overlay.png" alt="Segmentation">
                </div>
            </div>
            
            <div class="image-card">
                <div class="image-header">
                    <h3>Grain Boundaries</h3>
                    <div class="legend">
                        <div class="legend-item"><span class="dot" style="background: var(--success)"></span> Boundary</div>
                    </div>
                </div>
                <div class="img-container">
                    <img src="boundaries_overlay.png" alt="Boundaries">
                </div>
            </div>
        </div>
        
        <div class="details-section">
            <div class="section-header">
                <h2>Statistics</h2>
                <a href="grains.csv" class="download-btn">Download CSV Data</a>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Phase</th>
                        <th>Grain Count</th>
                        <th>Mean Area</th>
                        <th>Fraction</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="phase-badge badge-pearlite">Pearlite</span></td>
                        <td>{len(pearlite_stats):,}</td>
                        <td>{pearlite_stats['area_um2'].mean():.2f} µm²</td>
                        <td>{p_frac:.2f}%</td>
                    </tr>
                    <tr>
                        <td><span class="phase-badge badge-ferrite">Ferrite</span></td>
                        <td>{len(ferrite_stats):,}</td>
                        <td>{ferrite_stats['area_um2'].mean():.2f} µm²</td>
                        <td>{f_frac:.2f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""
        
        with open(f"{self.output_dir}/index.html", "w", encoding='utf-8') as f:
            f.write(html_content)
            
        return report_text
