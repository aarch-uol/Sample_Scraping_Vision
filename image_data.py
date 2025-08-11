class ImageData:
    """Class to hold all image processing data and intermediate results."""
    
    def __init__(self, color_image, depth_image, image_name):
        # Original inputs
        self.color_image = color_image # Original color image
        self.depth_image = depth_image # Original depth image
        self.image_name = image_name    # Name of the image file
        
        # Processing results (initialized as None)
        self.bounding_image = None # Color image with bounding box applied by YOLO
        self.grabcut_mask = None # Black and white mask created by GrabCut
        self.yolo_rect = None # Bounding box coordinates from YOLO detection
        self.extracted_depth = None # Depth values extracted using the GrabCut mask
        self.depth_denoised = None # Denoised version of the extracted_depth image
        self.thresh = None # Thresholded version of the depth_denoised image
        self.extracted_depth_colored = None # Colorized version of the extracted depth for visualization
        self.depth_denoised_colored = None # Colorized version of the denoised depth for visualization
        self.thresh_colored = None # Colorized version of the thresholded depth for visualization
        self.contourmap = None # Binary map created from the thresholded depth (black and white, white where crystals are)
        self.individual_contour_images = None # An array to hold individual contour images (each position in the array corresponds to a cluster of crystals)
        self.contoured_image = None # Final image with green contours drawn around detected crystals
        self.midpoint_coords = None # Midpoint coordinates of each detected contour
        self.spatula_results = None # Results from spatula extraction process