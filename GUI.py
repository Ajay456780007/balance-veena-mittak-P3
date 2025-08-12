import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model

# Placeholders for your feature extraction functions, adapt imports accordingly
def deep_color_based_pattern(img):
    # Your function code here
    # Use the same function code you provided above
    pass

def Deep_Structural_Pattern(img):
    # Your function code here
    pass

def glcm_statistical_features(img):
    # Your function code here
    pass

def Resnet151(img):
    # Your function code here
    pass

def deep_pixel_flow_map(img):
    # Your function code here
    pass

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Feature Extraction & Classification")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f2f5")

        # Model paths and loaded models dict
        self.models = {'DB1': None, 'DB2': None, 'DB3': None}
        self.model_paths = {'DB1': '', 'DB2': '', 'DB3': ''}
        self.current_model_key = None

        # Top frame for model buttons and path display
        top_frame = tk.Frame(root, bg="#f0f2f5")
        top_frame.pack(pady=15)

        for db_name in ['DB1', 'DB2', 'DB3']:
            btn = tk.Button(top_frame, text=db_name, width=8, font=("Helvetica", 12, "bold"),
                            command=lambda db=db_name: self.load_model_dialog(db))
            btn.pack(side=tk.LEFT, padx=10)

        # Model path display
        self.model_path_label = tk.Label(root, text="No model loaded", bg="#f0f2f5", fg="#555", font=("Helvetica", 10))
        self.model_path_label.pack(pady=5)

        # Image selection section
        img_frame = tk.Frame(root, bg="#f0f2f5")
        img_frame.pack(pady=10)

        select_img_btn = tk.Button(img_frame, text="Select Image", font=("Helvetica", 12), command=self.select_image)
        select_img_btn.pack()

        # Canvas to display loaded image
        self.img_canvas = tk.Canvas(root, width=224, height=224, bg="white", bd=2, relief=tk.RIDGE)
        self.img_canvas.pack(pady=10)

        # Predict button
        predict_btn = tk.Button(root, text="Predict", font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white",
                                command=self.predict)
        predict_btn.pack(pady=15)

        # Prediction result label
        self.result_label = tk.Label(root, text="", bg="#f0f2f5", font=("Helvetica", 14, "bold"))
        self.result_label.pack()

        # For storing the selected image
        self.selected_image = None

        # For dynamic classes (preset to 4, can be updated)
        self.class_labels = ["Class 1", "Class 2", "Class 3", "Class 4"]

    def load_model_dialog(self, db_key):
        path = filedialog.askopenfilename(title=f"Select model file for {db_key}",
                                          filetypes=[("Keras H5 Model", "*.h5"), ("All files", "*.*")])
        if path:
            try:
                self.models[db_key] = load_model(path)
                self.model_paths[db_key] = path
                self.current_model_key = db_key
                self.model_path_label.config(text=f"{db_key} model loaded from:\n{path}")
                messagebox.showinfo("Model Loaded", f"Model for {db_key} successfully loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def select_image(self):
        path = filedialog.askopenfilename(title="Select an image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if path:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Failed to load image.")
                return
            self.selected_image = img
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((224, 224), Image.ANTIALIAS)
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.img_canvas.delete("all")
            self.img_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
            self.result_label.config(text="")

    def predict(self):
        if self.current_model_key is None or self.models[self.current_model_key] is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
        if self.selected_image is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        # Extract features using your functions
        try:
            f1 = deep_color_based_pattern(self.selected_image)  # 150x150
            f2 = Deep_Structural_Pattern(self.selected_image)   # 150x150
            f3 = glcm_statistical_features(self.selected_image) # 150x150
            f4 = Resnet151(self.selected_image)                  # 150x150
            f5 = deep_pixel_flow_map(self.selected_image)        # 150x150
        except Exception as e:
            messagebox.showerror("Feature Extraction Error", str(e))
            return

        # Stack features channel-wise -> shape (150,150,5)
        features_stack = np.stack([f1, f2, f3, f4, f5], axis=-1)

        # Expand dims for batch
        features_batch = np.expand_dims(features_stack, axis=0)  # (1, 150, 150, 5)

        # Predict using the loaded model
        try:
            preds = self.models[self.current_model_key].predict(features_batch)
            pred_class = np.argmax(preds, axis=1)[0]
            pred_prob = preds[0][pred_class]

            # Display result
            label_text = f"Prediction: {self.class_labels[pred_class]} (Confidence: {pred_prob:.2f})"
            self.result_label.config(text=label_text)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
