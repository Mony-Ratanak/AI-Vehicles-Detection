from ultralytics import YOLO
import cv2
import os

# Define class names (ensure these match your dataset)
class_names = ['airplane', 'bicycles', 'cars', 'motorbikes', 'ships']
confidence_threshold = 0.9  # Set a confidence threshold

# Paths
output_dir = "images/predicted/"  # Directory to save output images
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Load the YOLO model (update the path to your trained weights)
model_path = "yolo-model.pt"
model = YOLO(model_path)

# Function to predict and process an image
def predict_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Perform prediction
    results = model.predict(source=img, save=False)

    # Extract the prediction results
    if len(results) > 0 and hasattr(results[0], "probs"):
        probs = results[0].probs
        predicted_class_idx = probs.top1  # Get the most likely class index
        confidence = probs.top1conf.item()  # Confidence for the top class

        # Determine class label and handle low confidence
        if confidence < confidence_threshold:
            predicted_class_name = "Not in class"
        else:
            if predicted_class_idx < len(class_names):
                predicted_class_name = class_names[predicted_class_idx]
            else:
                predicted_class_name = "Unknown class"

        # Display prediction
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}")

        # Annotate the image with the prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = f"Class: {predicted_class_name}, Conf: {confidence:.2f}"
        color = (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255)  # Green if valid, Red if "Not in class"
        cv2.putText(img, label_text, (10, 30), font, 0.8, color, 2)
    else:
        print(f"No valid predictions found for image: {image_path}")
        predicted_class_name = "No prediction"
        confidence = 0

    # Save the annotated image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Saved output image with prediction to: {output_path}\n")

# Loop to process images
while True:
    input_path = input("Enter the image path or directory (type '0' to exit): ").strip()

    # Exit condition
    if input_path == "0":
        print("Exiting the program.")
        break

    # Process all images in the directory or a single image
    if os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                predict_image(file_path)
    elif os.path.isfile(input_path):
        predict_image(input_path)
    else:
        print(f"Invalid path: {input_path}. Please enter a valid file or directory.")
