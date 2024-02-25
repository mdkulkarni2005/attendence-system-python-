import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime



video_capture = cv2.VideoCapture(1)

jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

manas_image = face_recognition.load_image_file("photos/manas.jpg")
manas_encoding = face_recognition.face_encodings(manas_image)[0]


known_face_encoding = [
    jobs_encoding,
    manas_encoding
]

known_face_names = [
    "jobs",
    "manas Kulkarni"
]

students = known_face_names.copy()


face_locations = []
face_encodings = []
face_names = []
s=True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f =open(current_date+ '.csv','w+', newline= '')
lnwriter = csv.writer(f)


while True:
    _,frame = video_capture.read()
    samll_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = samll_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            facce_distance = face_recognition.face_distance(known_face_encoding,face_encodings)
            best_match_index = np.argmin(facce_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
        cv2.imshow("attendence system", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()
