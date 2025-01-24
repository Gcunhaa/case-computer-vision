import cv2
import numpy as np
import easyocr

class LicencePlate:
    def __init__(self, box):
        self.box = box
        self.image = None
        self.processed_image = None
        self.segmented_characters = None
        self.extracted_text = None

    #TODO: Implement method for cropping the image

    def preprocess_image(self):
        if self.image is None:
            raise ValueError("Image is required for preprocessing")
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
        self.processed_image = cleaned
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

img = cv2.imread('plates/plate_0.jpg')
plate = LicencePlate(img)
plate.preprocess_image()
plate.segment_characters()
print(plate.extract_text())
cv2.waitKey(0)
cv2.destroyAllWindows()