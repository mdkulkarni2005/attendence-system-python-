import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Unable to open webcam")
    exit()

# Capture frames from the webcam
while True:
    ret, frame = cap.read()

    # Display the webcam frame
    cv2.imshow('Webcam', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()

# Now, import and run the attendance system

import cv2
import numpy as np
import csv
import os
import face_recognition
from datetime import datetime

# Load known faces
jobs_image = face_recognition.load_image_file("photos/jobs.jpeg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

manas_image = face_recognition.load_image_file("photos/manas.jpg")
manas_encoding = face_recognition.face_encodings(manas_image)[0]

known_face_encoding = [jobs_encoding, manas_encoding]
known_face_names = ["jobs", "manas Kulkarni"]

# Additional setup for attendance recording
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

students = known_face_names.copy()

# Create a VideoCapture object
video_capture = cv2.VideoCapture(0)  # Change the index according to your system (0 for default webcam)

# Check if the webcam is opened successfully
if not video_capture.isOpened():
    print("Unable to open webcam for attendance system")
    exit()

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Your face recognition code
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encodings)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Attendance recording
        if name in known_face_names:
            if name in students:
                students.remove(name)
                print(students)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    # Display the webcam frame for attendance system
    cv2.imshow("Attendance System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
f.close()
