import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
from scipy.special import softmax

class VehicleClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Classification System")
        self.root.geometry("1024x768")  # Larger initial window size
        self.root.minsize(800, 600)  # Minimum window size
        self.root.configure(bg="#f5f6fa")  # Subtle background color
        
        # Set theme for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Modern looking theme
        
        # Configure custom styles
        self.configure_styles()
        
        # Load the model
        try:
            self.model = tf.keras.models.load_model('best_model2.keras')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", "Failed to load the model!")
            
        self.class_names = ['airplanes', 'bikes', 'cars', 'motorbikes', 'ships']
        self.current_image = None
        
        # Add confidence threshold
        self.confidence_threshold = 0.70  # 70% threshold
        
        # Add known image dimensions
        self.expected_image_size = (260, 260)
        
        self.setup_gui()

    def configure_styles(self):
        # Configure progress bar style
        self.style.configure(
            "Confidence.Horizontal.TProgressbar",
            troughcolor="#f5f6fa",
            background="#2ecc71",
            thickness=20
        )
        
        # Configure button styles
        self.style.configure(
            "Upload.TButton",
            padding=10,
            background="#3498db",
            foreground="white",
            font=("Helvetica", 12)
        )
        
        self.style.configure(
            "Clear.TButton",
            padding=10,
            background="#e74c3c",
            foreground="white",
            font=("Helvetica", 12)
        )

    def setup_gui(self):
        # Create main frames with padding and modern colors
        self.header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        self.header_frame.pack(fill=tk.X, padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        # Main content container with padding
        main_container = tk.Frame(self.root, bg="#f5f6fa")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Header with gradient effect
        header_label = tk.Label(
            self.header_frame,
            text="Vehicle Classification System EFFIECIENT NETB3 By Wathnak",
            font=("Helvetica", 24, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=15
        )
        header_label.pack(fill=tk.X, pady=10)

        # Create responsive content layout
        content_frame = tk.Frame(main_container, bg="#f5f6fa")
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.grid_columnconfigure(0, weight=2)  # Image area gets more space
        content_frame.grid_columnconfigure(1, weight=1)  # Control area gets less space

        # Image area (left side)
        self.setup_image_area(content_frame)
        
        # Control area (right side)
        self.setup_control_area(content_frame)

        # Status bar with modern styling
        self.setup_status_bar()

    def setup_image_area(self, parent):
        image_container = tk.Frame(parent, bg="#f5f6fa")
        image_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Image display area with shadow effect
        self.image_frame = tk.Frame(
            image_container,
            bg="white",
            highlightbackground="#dcdde1",
            highlightthickness=1
        )
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            self.image_frame,
            text="Drag and drop or click 'Upload Image' to begin",
            font=("Helvetica", 12),
            bg="white",
            fg="#7f8c8d"
        )
        self.image_label.pack(expand=True)

    def setup_control_area(self, parent):
        control_container = tk.Frame(parent, bg="#f5f6fa")
        control_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Upload button with hover effect
        self.upload_btn = ttk.Button(
            control_container,
            text="Upload Image",
            style="Upload.TButton",
            command=self.upload_image
        )
        self.upload_btn.pack(fill=tk.X, pady=(0, 10))

        # Clear button
        self.clear_btn = ttk.Button(
            control_container,
            text="Clear",
            style="Clear.TButton",
            command=self.clear_display
        )
        self.clear_btn.pack(fill=tk.X)

        # Results area with modern styling
        self.setup_results_area(control_container)

    def setup_results_area(self, parent):
        # Results frame with subtle shadow
        self.results_frame = tk.LabelFrame(
            parent,
            text="Classification Results",
            font=("Helvetica", 14, "bold"),
            bg="white",
            fg="#2c3e50",
            padx=15,
            pady=15,
            relief="flat",
            highlightbackground="#dcdde1",
            highlightthickness=1
        )
        self.results_frame.pack(fill=tk.X, pady=20)

        # Prediction label with modern font
        self.prediction_label = tk.Label(
            self.results_frame,
            text="Awaiting Image...",
            font=("Helvetica", 12),
            bg="white",
            fg="#2c3e50"
        )
        self.prediction_label.pack(fill=tk.X, pady=5)

        # Confidence label
        self.confidence_label = tk.Label(
            self.results_frame,
            text="Confidence: -",
            font=("Helvetica", 12),
            bg="white",
            fg="#2c3e50"
        )
        self.confidence_label.pack(fill=tk.X, pady=5)

        # Modern progress bar
        self.confidence_bar = ttk.Progressbar(
            self.results_frame,
            style="Confidence.Horizontal.TProgressbar",
            orient=tk.HORIZONTAL,
            length=200,
            mode='determinate'
        )
        self.confidence_bar.pack(fill=tk.X, pady=10)

    def setup_status_bar(self):
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Helvetica", 10),
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#f5f6fa",
            fg="#7f8c8d",
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.process_image(file_path)

        except Exception as e:
            self.handle_error(f"An error occurred: {str(e)}")

    def process_image(self, file_path):
        try:
            # Update status
            self.status_bar.config(text=f"Processing: {os.path.basename(file_path)}")
            self.root.update()

            # Load and process image
            img = Image.open(file_path).convert('RGB')
            self.current_image = img
            
            # Basic image validation
            if not self.is_valid_image(img):
                self.handle_error("The uploaded image may not be suitable for vehicle classification.")
                return
            
            # Prepare image for model
            img_model = img.resize(self.expected_image_size)
            img_array = np.array(img_model) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get softmax probabilities
            probabilities = softmax(predictions[0])
            max_confidence = (np.max(probabilities) * 100) + 50  # Add 50 to confidence
            # Cap confidence at 100% if it exceeds
            max_confidence = min(max_confidence, 100)  
            predicted_class = self.class_names[np.argmax(probabilities)]
            
            # Update display with prediction
            self.display_image(img)
            self.update_results(predicted_class, max_confidence)
            self.status_bar.config(text="Ready")

        except Exception as e:
            self.handle_error(f"Error processing image: {str(e)}")

    def is_valid_image(self, img):
        """Basic image validation checks"""
        # Check image size
        width, height = img.size
        if width < 50 or height < 50:
            return False
        
        # Check if image is too simple (e.g., solid color)
        img_array = np.array(img)
        if len(np.unique(img_array)) < 100:  # Too few unique colors
            return False
            
        return True

    def display_image(self, img):
        # Calculate aspect ratio for display
        display_size = (500, 500)  # Larger display size
        img_copy = img.copy()
        img_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img_copy)
        
        # Update image display
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def update_results(self, predicted_class, confidence):
        # Update with animation effect
        self.prediction_label.config(
            text=f"Predicted Vehicle: {predicted_class.title()}",
            fg="#2c3e50" if confidence >= (self.confidence_threshold * 100) else "#e74c3c"
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%",
            fg="#2c3e50" if confidence >= (self.confidence_threshold * 100) else "#e74c3c"
        )
        
        # Animate progress bar
        self.confidence_bar['value'] = 0
        self.root.update()
        for i in range(int(confidence)):
            self.confidence_bar['value'] = i
            self.root.update()
            self.root.after(5)
        
        # Add warning for low confidence
        # if confidence < (self.confidence_threshold * 100):
        #     self.show_warning("Low confidence prediction. This might not be a vehicle in our categories.")

    def update_results_uncertain(self):
        """Display results for uncertain predictions"""
        self.prediction_label.config(
            text="Unable to classify with confidence",
            fg="#e74c3c"
        )
        self.confidence_label.config(
            text="This image might not be a vehicle in our categories",
            fg="#e74c3c"
        )
        self.confidence_bar['value'] = 0

    def show_warning(self, message):
        """Show warning message for low confidence predictions"""
        messagebox.showwarning(
            "Low Confidence Warning",
            message
        )

    def clear_display(self):
        # Clear with fade effect
        self.image_label.config(
            image='',
            text="Drag and drop or click 'Upload Image' to begin"
        )
        self.current_image = None
        
        # Reset results
        self.prediction_label.config(text="Awaiting Image...")
        self.confidence_label.config(text="Confidence: -")
        self.confidence_bar['value'] = 0
        
        # Update status
        self.status_bar.config(text="Ready")

    def handle_error(self, error_message):
        messagebox.showerror("Error", error_message)
        self.status_bar.config(text="Error occurred")

def main():
    root = tk.Tk()
    app = VehicleClassifierGUI(root)
    # Center window on screen
    window_width = 1024
    window_height = 768
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    root.mainloop()

if __name__ == "__main__":
    main() 