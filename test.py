import cv2
import face_recognition
import numpy as np


face_cap=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# enabling camera
video_cap=cv2.VideoCapture(0)

k_image = face_recognition.load_image_file(r"add the photo path")
k_face_encoding = face_recognition.face_encodings(k_image)[0]


known_face_encodings = [
    k_face_encoding,
]
known_face_names = [
    "add the name"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True:
    ret, video_data = video_cap.read()
    if process_this_frame:
        col = cv2.cvtColor(video_data,cv2.COLOR_BGR2RGB)
        face_locations=face_recognition.face_locations(col)
        face_encodings=face_recognition.face_encodings(col,face_locations)

        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
            name = "unkown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame

    
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(top, right, bottom, left), name in zip(face_locations,face_names):
        cv2.rectangle(video_data,(left,top),(left+top,right+bottom),(0,255,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(video_data, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow("live face detection --- press a to exit",video_data)
    
    if cv2.waitKey(10) == ord("a"):
        break
video_cap.release()


