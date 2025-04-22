import cv2
import cvzone
import math
import threading
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import pygame
# Initialize pygame to play the alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")  

# Load model YOLO
model = YOLO('yolov8s.pt')
classnames = model.names  # dictionary: {0: 'person', 1: 'bicycle', ...}

# GUI setup
root = tk.Tk()
root.title("Fall Detection System")
root.geometry("1000x800")

label_video = tk.Label(root)
label_video.pack()

label_warning = tk.Label(root, text="", font=("Arial", 32), fg="red")
label_warning.pack(pady=10)

status_bar = tk.Label(root, text="Following...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

cap = cv2.VideoCapture('fall.mp4')
prev_time = 0
fall_timer = 0

def process_frame():
    global prev_time, fall_timer

    ret, frame = cap.read()
    if not ret:
        status_bar.config(text="Video has ended or power error.")
        return

    frame = cv2.resize(frame, (980, 740))
    results = model(frame, verbose=False)

    fall_detected = False

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            class_name = classnames.get(cls_id, f"Unknown({cls_id})")
            confidence = math.ceil(conf * 100)

            width = x2 - x1
            height = y2 - y1
            is_fall = width > height  # if the person is lying horizontally

            if confidence > 80 and class_name == 'person':
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6, colorC=color)
                cvzone.putTextRect(frame, f'{class_name} {confidence}%', [x1 + 8, y1 - 12], thickness=2, scale=2)

                if is_fall:
                    fall_detected = True
                    cvzone.putTextRect(frame, ' FALL DETECTED ', [x1, y2 + 30], scale=2, thickness=2, colorR=(0, 0, 255))

    # FPS 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # Sound 
    if fall_detected:
        label_warning.config(text=" FALL DETECTED ")
        status_bar.config(text="Warning: Someone fell!")
        if time.time() - fall_timer > 5:  # play sound every 5 seconds
            pygame.mixer.Sound.play(alert_sound)
            fall_timer = time.time()
    else:
        label_warning.config(text="")
        status_bar.config(text="Following...")

    # Show image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    label_video.imgtk = img_tk
    label_video.configure(image=img_tk)

    root.after(10, process_frame)
# Start processing
threading.Thread(target=process_frame).start()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
