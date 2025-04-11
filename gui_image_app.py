import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os
import warnings

warnings.filterwarnings('ignore')

IMAGE_SIZE = (128, 128)

MODEL_SAVE_DIR = 'saved_image_models'
AGE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'age_model.h5')
GENDER_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'gender_model.h5')
HAIR_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'hair_model.h5')

AGE_CLASSES = ['lt_20', '20_30', 'gt_30']
GENDER_CLASSES = ['male', 'female']
HAIR_CLASSES = ['short', 'long']

TARGET_AGE_GROUP = '20_30'

try:
    print("Loading models...")
    age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
    gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
    hair_model = tf.keras.models.load_model(HAIR_MODEL_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

def preprocess_image(image_path, target_size):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image file {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        img_array = np.asarray(img_resized)
        img_normalized = img_array / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_image(image_path):
    global img_display

    status_label.config(text="Processing...")
    canvas.delete("all")
    root.update_idletasks()

    processed_image = preprocess_image(image_path, IMAGE_SIZE)
    if processed_image is None:
        status_label.config(text="Error: Could not process image.")
        return

    try:
        age_pred_prob = age_model.predict(processed_image)[0]
        age_pred_index = np.argmax(age_pred_prob)
        predicted_age_group = AGE_CLASSES[age_pred_index]
        print(f"Predicted Age Index: {age_pred_index}, Group: {predicted_age_group}, Prob: {age_pred_prob[age_pred_index]:.2f}")

        final_gender_prediction = "Unknown"

        if predicted_age_group == TARGET_AGE_GROUP:
            print("Applying Hair Length Rule (Age 20-30)")
            hair_pred_prob = hair_model.predict(processed_image)[0]
            hair_pred_index = np.argmax(hair_pred_prob)
            predicted_hair_length = HAIR_CLASSES[hair_pred_index]
            print(f"Predicted Hair Index: {hair_pred_index}, Length: {predicted_hair_length}, Prob: {hair_pred_prob[hair_pred_index]:.2f}")

            if predicted_hair_length == 'long':
                final_gender_prediction = 'Female (Rule: Long Hair)'
            elif predicted_hair_length == 'short':
                final_gender_prediction = 'Male (Rule: Short Hair)'
            else:
                final_gender_prediction = f"Unknown Hair ({predicted_hair_length})"

        else:
            print("Applying Standard Gender Rule (Age <20 or >30)")
            gender_pred_prob = gender_model.predict(processed_image)[0]
            if len(GENDER_CLASSES) == 2:
                pred_prob_female = gender_pred_prob[0]
                gender_pred_index = 1 if pred_prob_female > 0.5 else 0
                confidence = pred_prob_female if gender_pred_index == 1 else 1.0 - pred_prob_female
            else:
                gender_pred_index = np.argmax(gender_pred_prob)
                confidence = gender_pred_prob[gender_pred_index]

            predicted_gender = GENDER_CLASSES[gender_pred_index]
            print(f"Predicted Gender Index: {gender_pred_index}, Gender: {predicted_gender}, Conf: {confidence:.2f}")
            final_gender_prediction = f"{predicted_gender.capitalize()} (Rule: Standard Prediction)"

        result_text = f"Predicted Age Group: {predicted_age_group}\n"
        result_text += f"Final Predicted Gender: {final_gender_prediction}"
        status_label.config(text=result_text)

        img = Image.open(image_path)
        img.thumbnail((IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2))
        img_display = ImageTk.PhotoImage(img)
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, anchor=tk.CENTER, image=img_display)

    except Exception as e:
        status_label.config(text=f"An error occurred during prediction: {e}")
        print(f"Prediction error details: {e}")

def upload_action():
    try:
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if file_path:
            print(f"File selected: {file_path}")
            if not os.path.exists(file_path):
                status_label.config(text="Error: Selected file not found.")
                return
            predict_image(file_path)
        else:
            print("File selection cancelled.")
    except Exception as e:
        status_label.config(text=f"Error during file selection: {e}")
        print(f"File dialog error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Conditional Gender Identifier")
    root.geometry("500x550")

    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("ttk 'clam' theme not available, using default.")

    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(expand=True, fill=tk.BOTH)

    title_label = ttk.Label(main_frame, text="Image Analysis", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=(0, 10))

    upload_button = ttk.Button(main_frame, text="Upload Image File", command=upload_action, width=25)
    upload_button.pack(pady=10)

    canvas = tk.Canvas(main_frame, width=IMAGE_SIZE[0]*2, height=IMAGE_SIZE[1]*2, bg='lightgrey', relief=tk.SUNKEN, borderwidth=1)
    canvas.pack(pady=10)

    status_label = ttk.Label(
        main_frame,
        text="Upload an image to begin.",
        justify=tk.LEFT,
        wraplength=450,
        padding=(10, 10),
        relief=tk.GROOVE,
        borderwidth=1,
        anchor='nw',
        font=("Helvetica", 10)
    )
    status_label.pack(pady=10, fill=tk.X, expand=False)

    print("Starting GUI...")
    root.mainloop()
    print("GUI closed.")
