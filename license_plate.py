import cv2
import numpy as np
import easyocr

class LicencePlate:
    def __init__(self, box, confidence):
        self.box = box
        self.confidence = confidence
        self.image = None
        self.processed_image = None
        self.segmented_characters = None
        self.extracted_text = None

    def get_box_size(self):
        return (self.box[0] - self.box[2]) * (self.box[1] - self.box[3])

    def process_license_plate(self, frame):
        x1, y1, x2, y2 = map(int, self.box)
        plate_region = frame[y1:y2, x1:x2]
        self.image = plate_region
        self.preprocess_image()
        self.segment_characters()
        return self.extract_text()

    def preprocess_image(self):
        if self.image is None:
            raise ValueError("Image is required for preprocessing")
        
        # Store original image
        original = self.image.copy()
        
        # Resize image to 250px width while maintaining aspect ratio
        height, width = self.image.shape[:2]
        aspect_ratio = height / width
        new_width = 250
        new_height = int(new_width * aspect_ratio)
        self.image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Show original image
        cv2.imshow('Original Image', self.image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assumed to be the license plate)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Ensure correct orientation (width > height)
            if width < height:
                width, height = height, width
            
            # Get perspective transform
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                              [0, 0],
                              [width-1, 0],
                              [width-1, height-1]], dtype="float32")
            
            # Apply perspective transform
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.image = cv2.warpPerspective(self.image, matrix, (width, height))
            
            # Resize again to maintain consistent size
            self.image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Perspective Corrected', self.image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray)
        
        # Apply adaptive threshold
        thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
        cv2.imshow('Threshold', thresh_inv)
        
        # Apply morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Cleaned (Final)', cleaned)
        
        self.processed_image = cleaned
        
        # Wait for a key press to continue
        cv2.waitKey(0)
        return cleaned
    
    def segment_characters(self):
        if self.processed_image is None:
            raise ValueError("Processed image is required for segmentation")
     
        edges = self._canny_edge_detection(self.processed_image, 0.33)
        ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        img_area = self.image.shape[0]*self.image.shape[1]

        contour_img = self.image.copy()
        cv2.drawContours(contour_img, ctrs, -1, (0,255,0), 2)

        filtered_img = self.image.copy()
        char_images = []  # List to store individual character images

        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            roi_area = w*h
            roi_ratio = roi_area/img_area
            if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
                if ((h>1.2*w) and (3*w>=h)):
                    # Draw rectangle on filtered image
                    cv2.rectangle(filtered_img, (x,y), (x + w, y + h), (90,0,255), 2)
                    
                    # Add padding (2 pixels on each side)
                    padding = 2
                    y_start = max(0, y - padding)
                    y_end = min(self.processed_image.shape[0], y + h + padding)
                    x_start = max(0, x - padding)
                    x_end = min(self.processed_image.shape[1], x + w + padding)
                    
                    # Crop the character from the original image with padding
                    char_roi = self.processed_image[y_start:y_end, x_start:x_end]
                    
                    # Resize to 28x28 using cubic interpolation
                    char_roi_resized = cv2.resize(char_roi, (28, 28), interpolation=cv2.INTER_CUBIC)
                    char_images.append(char_roi_resized)

        # Display filtered image with rectangles
        cv2.imshow('Filtered Contours', filtered_img)
        self.segmented_characters = char_images
        # Display individual characters
        for i, char_img in enumerate(char_images):
            cv2.imshow(f'Character {i}', char_img)
            #cv2.imwrite(f'char_{i}.jpg', char_img)
        return char_images

    def extract_text(self):
        if self.segmented_characters is None:
            raise ValueError("Segmented characters are required for text extraction")
        
        reader = easyocr.Reader(['en'])
        text = []
        # Process each character image individually
        for char_img in self.segmented_characters:
            result = reader.readtext(char_img, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            if result:
                text.append(result[0])
        
        # Join all detected characters into a single string
        return ''.join(text) if text else 'None'

    def _canny_edge_detection(self, image, sigma):
        v = np.median(image)
    
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
    
        return edged


# img = cv2.imread('plates/plate_0.jpg')
# plate = LicencePlate(img)
# plate.preprocess_image()
# plate.segment_characters()
# print(plate.extract_text())
# cv2.waitKey(0)
# cv2.destroyAllWindows()