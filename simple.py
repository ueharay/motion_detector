import cv2

video = cv2.VideoCapture(1)
face_data = "haarcascade_frontalface_alt2.xml"
face_val = cv2.CascadeClassifier(face_data) #顔（cascade）を（face_data）から認識（ckassify）する機能、識別機みたいな
#顔認識機能
    

while True:
    check, frame = video.read()
    # このcheck,とframeに画像のデータを渡し、その画像を１枚ずつ無数に処理することで動画のように見せる
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # カメラの色をグレーにするコード
    detect_face = face_val.detectMultiScale(gray)
    #グレーの画像から顔を.detectデテクトする機能（face_valは顔認識情報が入った変数）※detectMultiSCaleは白黒しか認識不可=>gray変数を入れる

    for (x, y, w, h) in detect_face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #detecy_faceに上記の条件でrectangleを作る    

    cv2.imshow("Normal camera", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release 
# カメラを起動