import cv2
import numpy as np

class LicencePlate:
    def __init__(self, image):
        if self.image is None:
            raise ValueError("Image is required for creating a LicencePlate object")
        self.image = image
        self.processed_image = None
        self.segmented_characters = None
        self.extracted_text = None

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
                    
                    # Crop the character from the original image
                    char_roi = self.image[y:y+h, x:x+w]
                    char_images.append(char_roi)

        # Display filtered image with rectangles
        cv2.imshow('Filtered Contours', filtered_img)
        self.segmented_characters = char_images
        # Display individual characters
        for i, char_img in enumerate(char_images):
            cv2.imshow(f'Character {i}', char_img)

    def _canny_edge_detection(self, image, sigma):
        v = np.median(image)
    
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
    
        return edged

img = cv2.imread('plates/plate_0.jpg')
plate = LicencePlate(img)
plate.preprocess()
plate.segment_characters()
cv2.waitKey(0)
cv2.destroyAllWindows()