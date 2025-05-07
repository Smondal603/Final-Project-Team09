def process_image(image_path, original_name=None):
    """Process a single image and return metrics dictionary"""
    metrics = {"Image": os.path.basename(image_path) if original_name is None else original_name}

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        width, height = image.size

        # Crop to square
        min_dim = min(width, height)
        if width < height:
            top = (height - min_dim) // 2
            bottom = top + min_dim
            left, right = 0, min_dim
        else:
            left = (width - min_dim) // 2
            right = left + min_dim
            top, bottom = 0, min_dim

        image = image.crop((left, top, right, bottom))
        image_array = np.array(image)

        # Image processing
        A = np.array(image)
        B = np.ones_like(A) * 255
        C = B - A

        # Thresholding
        _, thresholded = cv2.threshold(C, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Contour detection
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No contours found")

        # Separate contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        central_line_contour = contours[0]
        overspray_contours = contours[1:] if len(contours) > 1 else []

        # Create mask images
        central_line_image = np.zeros_like(thresholded)
        cv2.drawContours(central_line_image, [central_line_contour], -1, 255, thickness=cv2.FILLED)
        overspray_image = np.zeros_like(thresholded)
        cv2.drawContours(overspray_image, overspray_contours, -1, 255, thickness=cv2.FILLED)

        # Calculate metrics
        central_line_points = central_line_contour[:, 0, :]
        y_coords = np.unique(central_line_points[:, 1])
        widths = [
            np.ptp(central_line_points[central_line_points[:, 1] == y, 0])
            for y in y_coords
            if len(central_line_points[central_line_points[:, 1] == y, 0]) > 1
        ]
        average_line_width = np.mean(widths) if widths else np.nan

        # Edge roughness calculation
        M = cv2.moments(central_line_contour)
        centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        edges = cv2.Canny(central_line_image, 100, 200)
        edge_coords = np.column_stack(np.where(edges > 0))

        left_edge_coords = edge_coords[edge_coords[:, 1] < centroid_x]
        right_edge_coords = edge_coords[edge_coords[:, 1] >= centroid_x]

        nominal_line_left = np.mean(left_edge_coords[:, 1]) if left_edge_coords.size > 0 else np.nan
        nominal_line_right = np.mean(right_edge_coords[:, 1]) if right_edge_coords.size > 0 else np.nan

        distances_left = left_edge_coords[:, 1] - nominal_line_left if left_edge_coords.size > 0 else np.array([])
        distances_right = right_edge_coords[:, 1] - nominal_line_right if right_edge_coords.size > 0 else np.array([])

        rms_edge_roughness = np.sqrt(np.mean(np.concatenate((distances_left, distances_right)) ** 2)) if distances_left.size + distances_right.size > 0 else np.nan

        # Additional metrics
        uniformity = np.std(widths) if widths else np.nan
        central_line_area = np.sum(central_line_image > 0)
        overspray_area = np.sum(overspray_image > 0)
        total_area = thresholded.shape[0] * thresholded.shape[1]

        overspray_percentage = (overspray_area / (total_area - central_line_area)) * 100 if central_line_area < total_area else np.nan
        overspray_ratio_to_lw = (overspray_area / central_line_area) * 100 if central_line_area > 0 else np.nan

        x_coords = np.unique(central_line_points[:, 0])
        max_line_length = np.max([
            np.ptp(central_line_points[central_line_points[:, 0] == x, 1])
            for x in x_coords
            if len(central_line_points[central_line_points[:, 0] == x, 1]) > 1
        ]) if x_coords.size else np.nan

        total_line_pixels = np.sum(central_line_image > 0)
        discontinuous_pixels = total_line_pixels - max_line_length
        discontinuity_percent = (discontinuous_pixels / total_line_pixels) * 100 if total_line_pixels > 0 else np.nan

        # Overspray statistics
        overspray_areas = [cv2.contourArea(c) for c in overspray_contours]
        mean_overspray_area = np.mean(overspray_areas) if overspray_areas else np.nan
        std_overspray_area = np.std(overspray_areas) if overspray_areas else np.nan

        # Convert overspray area to μm²
        mean_overspray_area_um2 = mean_overspray_area * (pixel_to_micron ** 2) if not np.isnan(mean_overspray_area) else np.nan
        std_overspray_area_um2 = std_overspray_area * (pixel_to_micron ** 2) if not np.isnan(std_overspray_area) else np.nan

        center_x_px = np.mean(central_line_points[:, 0])
        num_overspray_left = sum(1 for c in overspray_contours if np.mean(c[:, 0, 0]) < center_x_px)
        num_overspray_right = sum(1 for c in overspray_contours if np.mean(c[:, 0, 0]) > center_x_px)
        total_num_overspray = len(overspray_contours)

        # Final metrics assembly
        metrics.update({
            "Line Type": "Continuous" if max_line_length >= 0.95 * min_dim else "Discontinuous",
            "Average Line Width (px)": average_line_width,
            "Average Line Width (μm)": average_line_width * pixel_to_micron,
            "RMS Edge Roughness (px)": rms_edge_roughness,
            "RMS Edge Roughness (μm)": rms_edge_roughness * pixel_to_micron,
            "Uniformity (px)": uniformity,
            "Uniformity (μm)": uniformity * pixel_to_micron,
            "Overspray Percentage (%)": overspray_percentage,
            "Overspray Ratio to LW (%)": overspray_ratio_to_lw,
            "Vertical Max Pixel Value": max_line_length,
            "Continuity Percentage (%)": discontinuity_percent,
            "Mean Overspray Area (px²)": mean_overspray_area,
            "Std. Overspray Area (px²)": std_overspray_area,
            "Mean Overspray Area (μm²)": mean_overspray_area_um2,
            "Std. Overspray Area (μm²)": std_overspray_area_um2,
            "Count of Overspray (Left)": num_overspray_left,
            "Count of Overspray (Right)": num_overspray_right,
            "Total Number of Overspray": total_num_overspray,
        })

    except Exception as e:
        metrics["Error"] = str(e)

    return metrics 