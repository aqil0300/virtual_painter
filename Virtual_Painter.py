import cv2
import HandTrackingModule as htm
import numpy as np
import time



detector = htm.handDetector()

draw_color = (0,0,0)

img_canvas = np.zeros((580,1060,3),np.uint8)

prev_time = 0


cap = cv2.VideoCapture(0)


while True:

    success,frame = cap.read()

    frame = cv2.resize(frame,(1060,580))
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame,(10,10),(200,100),(255,0,0),cv2.FILLED)
    cv2.rectangle(frame,(205,10),(400,100),(0,0,255),cv2.FILLED)
    cv2.rectangle(frame,(405,10),(600,100),(0,255,200),cv2.FILLED)
    cv2.rectangle(frame,(605,10),(800,100),(255,0,255),cv2.FILLED)
    cv2.rectangle(frame,(805,10),(1050,100),(255,255,255),cv2.FILLED)
    cv2.putText(frame,text='Erase',org=(860,66),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=3,color=(0,0,0),thickness=5)

#find hand landmarks

    frame= detector.findHands(frame,draw=True)
    lmlist = detector.findPosition(frame)


    #print(lmlist)
    if len(lmlist)!=0:
    
       x1,y1 = lmlist[8][1:]     #index finger coordinates
       x2,y2 = lmlist[12][1:]    #middle finger coordinates
    
      # print(x1,y1)

#check which finger is up

       fingers = detector.fingersUp()

       #print(fingers)

#selection mode - two finger is up

       if fingers [1] and fingers [2]:
           
           xp,yp = 0,0
           
           #print('Selection mode')

           if y1 <100:
               
               if 10<x1<230:
                   draw_color = (255,0,0)
                  # print('blue')
                
               elif 240<x1<460:
                   draw_color = (0,0,255)
                   #print('red')
               
               elif 470<x1<690:
                   draw_color = (0,255,200)
                   #print('yellow')

               elif 700<x1<920:
                   draw_color = (255,0,255)
                   #print('pink')

               elif 930<x1<1970:
                   draw_color = (0,0,0)
                   #print('eraser')    




           cv2.rectangle(frame,(x1,y1),(x2,y2),draw_color,thickness=cv2.FILLED)
#drawing mode - index finger is up

       if (fingers[1] and not fingers[2]):
           #print('drawing mode')

           cv2.putText(frame,'drawing mode',(800,450),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(255,69,0),fontScale=1,thickness=3)
           cv2.circle(frame,(x1,y1),10,draw_color,thickness=-1)



           if xp == 0 and yp == 0:
               xp = x1
               yp = y1



           if draw_color == (0,0,0):
                cv2.line(frame,(xp,yp),(x1,y1),color=draw_color,thickness=10)
                cv2.line(img_canvas,(xp,yp),(x1,y1),color=draw_color,thickness=10)

           else:
            cv2.line(frame,(xp,yp),(x1,y1),color=draw_color,thickness=10)
            cv2.line(img_canvas,(xp,yp),(x1,y1),color=draw_color,thickness=10)



           xp,yp = x1,y1


# merging 2 canvas
    img_grey = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)

# thresh inverse

    thresh,img_inv = cv2.threshold(img_grey,20,255,cv2.THRESH_BINARY_INV) 

    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)




#AND Operation

    frame = cv2.bitwise_and(frame,img_inv)
    frame = cv2.bitwise_or(frame,img_canvas)

    frame = cv2.addWeighted(frame,1,img_canvas,0,5,0)

#calculating tips

    c_time = time.time()

    fps = 1 / (c_time - prev_time)

    prev_time = c_time

    cv2.putText(frame,str(int(fps)),(50,180),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),thickness=3)



    cv2.imshow("virtual_painting", frame)
    #cv2.imshow('canvas',img_canvas)
    #cv2.imshow('grey',img_inv)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
