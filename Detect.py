#!/usr/bin/env python3

import cv2
from datetime import datetime
import operator
import cognitive_face as CF

#all_attributes = 'age,gender,headPose,smile,facialHair,glasses,emotion'
attributes = 'smile,emotion'
emotionItems = ['anger','contempt','disgust','fear','happiness','neutral','sadness' ,'surprise']

def detectFaces():
    lastPersons = []
    result = CF.face.detect('currentImage.jpg', True, False, attributes)
    print(result)
    list = []
    dict ={}
    for item in result :
        list.append(item['faceId'])
        stats = item['faceAttributes']['emotion']
        maxItem = stats['anger']
        maxEmo = 'anger'
        for m in emotionItems:
            if maxItem<stats[m]:
                maxItem=stats[m]
                maxEmo=m
        dict[item['faceId']] = maxEmo;
        if len(list) > 0:
            result = CF.face.identify(list,'usergroup')
            for item in result :
                for inner in item['candidates']:
                    #print(item)
                    result = CF.person.get('usergroup', inner['personId'])
                    lastPersons.append("{} is {} emotion".format(result['name'],dict[item['faceId']]) )
        else :
            lastPersons.append("No Faces Found!!!" )
    return lastPersons
def main():
    KEY = '677a9fa1bda549a7a0dee3b90bba4a07'  # Replace with a your Subscription Key here.
    CF.Key.set(KEY)

    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    cap = cv2.VideoCapture(0)
    print("Capturing Image at {} x {} size".format(cap.get(3),cap.get(4)))
    
    webcam.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    webcam.set(CV_CAP_PROP_FRAME_WIDTH ,1280);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT ,960);
    
    lastPersons = []
    while(True): 
        c = cv2.waitKey(1);
    
        if 'q' == chr(c & 255):
            break
        
        # Capture frame-by-frame 
        ret, frame = cap.read()
        if 'd' == chr(c & 255):
            cv2.imwrite('currentImage.jpg', frame.copy())
        overlay = cv2.flip(frame.copy(),1)
        cv2.rectangle(overlay,(50,50),(350,250),(0,0,0),-1)
        cv2.putText(overlay, 'ZenLabs',(75,75),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(255,255,255))
        cv2.putText(overlay, str(datetime.now()),(75,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(255,255,255))
    
        if 'd' == chr(c & 255):
            cv2.putText(overlay, 'Person detection in Progress',(75,125),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(255,255,255))
        else:
            cv2.putText(overlay, 'Press d to Detect User',(75,125),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(255,255,255))
        if len(lastPersons) > 0:
            cv2.putText(overlay, 'Last Detected Persons',(75,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(255,255,255))
            lc = 0
            for item in lastPersons:
                cv2.putText(overlay, item,(75,175+lc),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(255,255,255))
                lc = lc +25     
        output = cv2.flip(frame.copy(),1)
        gray = cv2.cvtColor(output.copy(), cv2.COLOR_BGR2GRAY)    
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.5, output, 0.5,0, output)
        # Our operations on the frame come here
        #r = 1000/output.shape[1]
        #dim = (1000,int(r*output.shape[0]))
        #resized_img = cv2.resize(output,dim,interpolation = cv2.INTER_AREA)
        # Display the resulting frame
        cv2.namedWindow("VisionDemo", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("VisionDemo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("VisionDemo", output)
        if 'd' == chr(c & 255):
            lastPersons = detectFaces()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()