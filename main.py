import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st
st.title("ACCESS AUTHORIZATION SYSTEM")
st.header("CLICK TO OPEN WEBCAM")
run = st.checkbox('click')
FRAME_WINDOW = st.image([])
path ='images'
images=[]
personName=[]
myList=os.listdir(path)
print(myList)
for cu_img in myList:
    current_img=cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)

def faceEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown=faceEncodings(images)
print("all endoing done")


def attendance(name):
    with open('Attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        
        if name not in namelist:
            time_now=datetime.now()
            tstr=time_now.strftime('%H:%M:%S')
            dstr=time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tstr},{dstr}')
    
camera=cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodeCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
           # cv2.putText(frame, , (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            attendance(name)
            if cv2.waitKey(10)==13:
                break
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, "not permitted", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        

    FRAME_WINDOW.image(frame)
    # print("permission granted")  

else:
    st.write('Stopped')

# while True:
#     ret,frame=cap.read()
#     faces=cv2.resize(frame,(0,0),None,0.25,0.25)
#     faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    
#     facescurrentframe=face_recognition.face_locations(faces)
#     encodescurrentfframe=face_recognition.face_encodings(faces,facescurrentframe)
    
#     for encodeFace, faceloc in zip(encodescurrentfframe,facescurrentframe):
#         matches=face_recognition.compare_faces(encodelistknown,encodeFace)
#         facedis=face_recognition.face_distance(encodelistknown,encodeFace)
        
#         matchindex=np.argmin(facedis)
        
#         if matches[matchindex]:
#             name=personName[matchindex].upper()
#             #print(name)
#             y1,x2,y2,x1=faceloc
#             y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            
#     cv2.imshow("camera", frame)
    camera.release()

# cv2.destroyAllWindows()
