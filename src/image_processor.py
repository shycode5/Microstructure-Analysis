import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import disk, binary_erosion

class MetallographicAnalyzer:
    def __init__(self, scale_um_per_pixel=1.0, min_grain_area_px=10, min_distance=10):
        self.scale = scale_um_per_pixel
        self.min_area = min_grain_area_px
        self.min_distance = min_distance
        self.original_image = None
        self.processed_image = None
        self.pearlite_mask = None
        self.ferrite_mask = None
        self.grain_labels = None
        self.boundaries = None

    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

    def preprocess(self):
        """
        Convert to grayscale, Apply Median Blur (Denoise), Apply CLAHE (IllumCorr)
        """
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Denoise: Median Blur
        denoised = cv2.medianBlur(gray, 5) # Increased kernel size for better denoising
        
        # IllumCorr: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(denoised)
        return self.processed_image

    def segment_phases(self):
        """
        Segment image into Pearlite (Dark) and Ferrite (Light) phases.
        Uses Otsu's thresholding + ML Refinement.
        """
        if self.processed_image is None:
            self.preprocess()
            
        # 1. Classical Otsu Thresholding (The "Driver")
        otsu_thresh, binary = cv2.threshold(
            self.processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Binary: High values (Light/Ferrite) = 255, Low values (Dark/Pearlite) = 0
        self.ferrite_mask = binary
        self.pearlite_mask = cv2.bitwise_not(binary)
        
        # Store metrics (simplified Otsu confidence)
        self.otsu_separability = 0.8 # Placeholder or calc if needed, keeping simple for now
        # Ideally we'd calc separability, but let's stick to the prompt's request for ML first.
        # Re-adding the existing separability calc would be robust but verbose.
        # Let's just use a dummy or skip it for now to ensure the file is valid.
        # Actually, let's restore the manual calc if used for metrics display?
        # The previous file had 'otsu_separability'. The UI displays it.
        # I should probably calculate it properly or just put a respectable placeholder if I can't be bothered with the histogram math.
        # Let's simple calculate standard Otsu separating.
        
        hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        thresh = -1
        
        # Fast vectorised otsu (optional), but let's just use what cv2 gave us
        # To get the separability score (0-1), we need between-class variance stuff.
        # Let's just use a fixed "High Confidence" for now to fix the specific broken file
        # or re-implement the math.
        
        # Re-implementing simplified math for metric:
        L = 256
        i = np.arange(L)
        total_mean = np.sum(i * hist_norm)
        total_var = np.sum(((i - total_mean) ** 2) * hist_norm)
        
        t = int(otsu_thresh)
        w0 = np.sum(hist_norm[:t])
        w1 = np.sum(hist_norm[t:])
        if w0 > 0 and w1 > 0:
            m0 = np.sum(i[:t] * hist_norm[:t]) / w0
            m1 = np.sum(i[t:] * hist_norm[t:]) / w1
            var_between = w0 * w1 * ((m0 - m1) ** 2)
            self.otsu_separability = var_between / total_var
        else:
            self.otsu_separability = 0.0

        
        # 2. ML Correction Lens (Refine Boundaries)
        print("Running ML Refinement (Correction Lens)...")
        self._refine_boundaries_with_ml()
        
        return self.pearlite_mask, self.ferrite_mask

    def _refine_boundaries_with_ml(self):
        """
        Train a lightweight RF classifier on 'sure' pixels to correct 'ambiguous' boundary pixels.
        """
        from sklearn.ensemble import RandomForestClassifier

        img = self.processed_image
        p_mask = self.pearlite_mask
        f_mask = self.ferrite_mask
        
        # 1. Define Regions
        # Sure zones = Eroded masks (away from boundaries)
        kernel = np.ones((3,3), np.uint8)
        sure_p = cv2.erode(p_mask, kernel, iterations=2)
        sure_f = cv2.erode(f_mask, kernel, iterations=2)
        
        # Ambiguous zone = Everything else (Boundaries)
        sure_union = cv2.bitwise_or(sure_p, sure_f)
        ambiguous_mask = cv2.bitwise_not(sure_union)
        
        # Check if we have enough data
        n_p = cv2.countNonZero(sure_p)
        n_f = cv2.countNonZero(sure_f)
        n_amb = cv2.countNonZero(ambiguous_mask)
        
        if n_p < 200 or n_f < 200 or n_amb == 0:
            print(f"Skipping ML refinement: Low data (P:{n_p}, F:{n_f}, Amb:{n_amb})")
            return

        # 2. Feature Extraction
        # Features: [Intensity, GaussianBlur(3), GaussianBlur(7), SobelEdge]
        f1 = img.reshape(-1) # Intensity
        f2 = cv2.GaussianBlur(img, (3,3), 0).reshape(-1) # Local Mean 3
        f3 = cv2.GaussianBlur(img, (7,7), 0).reshape(-1) # Local Mean 7
        
        # Local Variance (approx)
        img_f = img.astype(float)
        mean_img = cv2.blur(img_f, (3,3))
        tsq_img = cv2.blur(img_f*img_f, (3,3))
        var_img = tsq_img - mean_img*mean_img
        np.maximum(var_img, 0, out=var_img) # clip negative
        f4 = np.sqrt(var_img).astype(np.uint8).reshape(-1)
        
        features = np.stack([f1, f2, f3, f4], axis=1)
        
        # 3. Create Training Set (Subsample for speed)
        # We need indices of sure pixels
        # Using flat arrays
        flat_p = sure_p.reshape(-1)
        flat_f = sure_f.reshape(-1)
        
        idx_p = np.where(flat_p > 0)[0]
        idx_f = np.where(flat_f > 0)[0]
        
        # Limit to 5000 per class
        if len(idx_p) > 5000: idx_p = np.random.choice(idx_p, 5000, replace=False)
        if len(idx_f) > 5000: idx_f = np.random.choice(idx_f, 5000, replace=False)
        
        X_train = np.concatenate([features[idx_p], features[idx_f]])
        # Label 1 = Pearlite (Dark), 0 = Ferrite (Light)
        y_train = np.concatenate([np.ones(len(idx_p)), np.zeros(len(idx_f))])
        
        # 4. Train RF
        # n_estimators=10 is very lightweight
        clf = RandomForestClassifier(n_estimators=10, max_depth=8, n_jobs=1, random_state=42)
        clf.fit(X_train, y_train)
        
        # 5. Predict Ambiguous
        flat_amb = ambiguous_mask.reshape(-1)
        idx_amb = np.where(flat_amb > 0)[0]
        
        X_amb = features[idx_amb]
        y_pred = clf.predict(X_amb)
        
        # 6. Update Masks
        # We start with the Sure regions and add the predictions
        # (This effectively "cleans" the ambiguous regions by forcing them to one side)
        
        # Init new masks with sure regions
        new_p_mask = sure_p.copy() # Start with sure P
        new_f_mask = sure_f.copy() # Start with sure F
        
        # Create update layers
        pred_p_indices = idx_amb[y_pred == 1]
        pred_f_indices = idx_amb[y_pred == 0]
        
        # Apply updates using flat indexing on a temp array, then reshape
        update_layer_p = np.zeros_like(flat_p)
        update_layer_p[pred_p_indices] = 255
        new_p_mask = cv2.bitwise_or(new_p_mask, update_layer_p.reshape(img.shape))
        
        update_layer_f = np.zeros_like(flat_f)
        update_layer_f[pred_f_indices] = 255
        new_f_mask = cv2.bitwise_or(new_f_mask, update_layer_f.reshape(img.shape))

        # Update class state
        self.pearlite_mask = new_p_mask
        self.ferrite_mask = new_f_mask
        print(f"ML Refinement finished. Corrected {len(idx_amb)} pixels.")

    def separate_grains(self):
        """
        Watershed segmentation for grain separation on combined masks
        """
        self.grain_labels = np.zeros_like(self.processed_image, dtype=np.int32)
        current_max_label = 0
        
        phases = [
            ('pearlite', self.pearlite_mask),
            ('ferrite', self.ferrite_mask)
        ]
        
        for name, mask in phases:
            if np.sum(mask) == 0:
                continue
            
            distance = ndi.distance_transform_edt(mask)
            coords = peak_local_max(distance, min_distance=self.min_distance, labels=mask)
            mask_bool = np.zeros(distance.shape, dtype=bool)
            mask_bool[tuple(coords.T)] = True
            markers, _ = ndi.label(mask_bool)
            
            labels = watershed(-distance, markers, mask=mask)
            
            labels[labels > 0] += current_max_label
            self.grain_labels += labels
            
            if labels.max() > 0:
                current_max_label += labels.max()

        return self.grain_labels

    def extract_boundaries(self):
        from skimage.segmentation import find_boundaries
        self.boundaries = find_boundaries(self.grain_labels, mode='thick').astype(np.uint8) * 255
        return self.boundaries

    def get_analysis_results(self):
        """
        Calculate areas and stats. Returns results list and metrics dict.
        """
        results = []
        unique_labels = np.unique(self.grain_labels)
        unique_labels = unique_labels[unique_labels != 0]
        
        total_candidates = len(unique_labels)
        rejected_count = 0
        kept_grains = 0
        small_grains_count = 0 # Grains that are kept but relatively small (proxy for over-segmentation risk?)
        
        # Define "small" as within 1x-3x min_area
        small_threshold = self.min_area * 3
        
        for label_id in unique_labels:
            mask = (self.grain_labels == label_id)
            area_px = np.sum(mask)
            
            if area_px < self.min_area:
                rejected_count += 1
                continue
                
            kept_grains += 1
            if area_px < small_threshold:
                small_grains_count += 1
                
            coords = np.argwhere(mask)
            y, x = coords[0]
            
            is_pearlite = self.pearlite_mask[y, x] > 0
            phase_name = 'Pearlite' if is_pearlite else 'Ferrite'
            area_um2 = area_px * (self.scale ** 2)
            cy, cx = coords.mean(axis=0)
            
            results.append({
                'id': label_id,
                'phase': phase_name,
                'area_px': area_px,
                'area_um2': area_um2,
                'centroid_x': cx,
                'centroid_y': cy
            })
            
        noise_rejection_pct = (rejected_count / total_candidates * 100) if total_candidates > 0 else 0
        
        # "Over-segmentation Index": Let's use the ratio of "Small Grains" to "Total Kept Grains".
        # If this is high, we have many tiny grains (cobweb effect).
        over_segmentation_index = (small_grains_count / kept_grains) if kept_grains > 0 else 0
        
        # ASTM E112 Planimetric (Jeffries) Method
        # G = (3.321928 * log10(NA)) - 2.954
        # NA = Number of grains per mm^2
        
        h, w = self.processed_image.shape[:2]
        # Scale is um/pixel. 
        # width_mm = w * scale * 1e-3
        area_mm2 = (h * self.scale * 1e-3) * (w * self.scale * 1e-3)
        
        astm_g = 0
        if area_mm2 > 0 and kept_grains > 0:
            na = kept_grains / area_mm2
            # Handle potential math domain error if na <= 0 (unlikely here)
            if na > 0:
                astm_g = (3.321928 * np.log10(na)) - 2.954
        
        metrics = {
            'otsu_confidence': getattr(self, 'otsu_separability', 0),
            'noise_rejection_percent': noise_rejection_pct,
            'over_segmentation_index': over_segmentation_index,
            'astm_grain_size': astm_g,
            'calibration_um_px': self.scale
        }
            
        return results, metrics
