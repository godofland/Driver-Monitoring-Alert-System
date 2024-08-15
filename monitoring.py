import cv2
from playsound import playsound
import threading

# Load the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to keep track of drowsiness
closed_eyes_frame_count = 0
drowsiness_threshold = 20  # Adjust this value based on testing
alarm_on = False

def play_alarm():
    playsound('alarm.wav')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            closed_eyes_frame_count += 1
        else:
            closed_eyes_frame_count = 0
            alarm_on = False  # Reset alarm flag if eyes are open

        # Check if eyes are closed for a certain number of frames
        if closed_eyes_frame_count >= drowsiness_threshold and not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm).start()

    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
