from ultralytics import YOLO
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ocr import extract_text_from_plate
from preprocess import preprocess_plate
# Load the model
model = YOLO("best.pt")

COUNTER = 0
SAVED_PLATES = []  # Store previously saved plates for comparison

# def are_plates_similar(plate1, plate2, threshold=0.4):
#     # Resize plates to same size for comparison
#     size = (100, 50)  # arbitrary size that maintains aspect ratio
#     plate1 = cv2.resize(plate1, size)
#     plate2 = cv2.resize(plate2, size)
    
#     # Convert to grayscale
#     plate1_gray = cv2.cvtColor(plate1, cv2.COLOR_BGR2GRAY)
#     plate2_gray = cv2.cvtColor(plate2, cv2.COLOR_BGR2GRAY)
    
#     # Compare using SSIM
#     score, _ = ssim(plate1_gray, plate2_gray, full=True)
#     return score > threshold

def predict_frame(frame):
    global COUNTER, SAVED_PLATES
    
    # Run inference on the frame
    results = model(frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("Case Gabriel", annotated_frame)
    
    # Get the first (and only) result for this frame
    result = results[0]
    boxes = result.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        plate_region = frame[y1:y2, x1:x2]
        
        # Check if this plate is similar to any we've saved before
        is_duplicate = False
        #for saved_plate in SAVED_PLATES:
            # if are_plates_similar(plate_region, saved_plate):
            #     is_duplicate = True
            #     break
        
        # Only save if it's not a duplicate
        if not is_duplicate:
            pre_processed_plate = preprocess_plate(plate_region)
            extracted_text = extract_text_from_plate(pre_processed_plate)
            print(extracted_text)
            cv2.imwrite(f'plates/plate_{extracted_text}.jpg', pre_processed_plate)
            SAVED_PLATES.append(pre_processed_plate)

            COUNTER += 1
            
    return annotated_frame



# Function to process video
def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference on the frame
            predict_frame(frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the video is finished
            break
    
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)
    
    if frame is not None:
        # Run YOLOv8 inference on the frame
        predict_frame(frame)
        
        # Wait for a key press
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not read image at {image_path}")

# Example usage
video_path = "datasets/video-india.mp4"  # Replace with your video path
#process_video(video_path)
process_image("datasets/mini.jpeg")
