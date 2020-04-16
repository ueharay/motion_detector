import cv2, pandas as pd
from datetime import datetime

video = cv2.VideoCapture(1)
front_face = "haarcascade_frontalface_alt2.xml"
eye = "haarcascade_eye.xml"

face_val = cv2.CascadeClassifier(front_face)
eye_val = cv2.CascadeClassifier(eye)
#顔（cascade）を（face_data）から認識（ckassify）する機能、識別機みたいな
status_log = [None, None]
time = [None]
df = pd.DataFrame(columns=["Start", "End"])

while True:
    status = 0 
    check, frame = video.read()
    # このcheck,とframeに画像のデータを渡し、その画像を１枚ずつ無数に処理することで動画のように見せる
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # カメラの色をグレーにするコード
    detect_face = face_val.detectMultiScale(gray)
    detect_eye = eye_val.detectMultiScale(gray)
    #グレーの画像から顔を.detectデテクトする機能（face_valは顔認識情報が入った変数）※detectMultiSCaleは白黒しか認識不可=>gray変数を入れる
    for (x, y, w, h) in detect_face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        for (ex, ey, ew, eh) in detect_eye:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
    #detect_faceに上記の条件でrectangleを作る    
        status = 1
        continue
    status_log.append(status)

    if status_log[-1] == 1 and status_log[-2] == 0:
        time.append(datetime.now())  
    if status_log[-1] == 0 and status_log[-2] == 1:
        time.append(datetime.now())            
    #[]リストやから次々と0, 1が[1,1,1,0,1,0,0,0,0]みたいな感じで足されていって、-1(最後のindex)が1(移ってる)
    #で-2(最後から２番目のindex)が0(何も映っていない)だったらtime.append時間をするというコード
    #つまり、何もいない所から誰か映るか、その逆が起こったらtime.append(datetime.now())するってこと

    cv2.imshow("Normal camera", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        if status_log == 1:
            time.append(datetime.now())
        break

for i in range(0, len(time), 2):
    df = df.append({"Start":time[i], "End":time[i+1]}, ignore_index=True)
    df.to_csv("time_log.csv")    

video.release 
# カメラを終了