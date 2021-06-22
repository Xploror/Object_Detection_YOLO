import cv2
import numpy as np
import time

ptime = 0
ctime = 0
net = cv2.dnn.readNet('yolov3_last.weights', 'yolov3_custom.cfg')
classes = []
font = cv2.FONT_HERSHEY_PLAIN
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter('output1.mp4', fourcc, 30.0, (352,640))

with open('obj.names', 'r') as f:
    classes = f.read().splitlines()
    
print(len(classes))

#img = cv2.imread('office.jfif')
#cap = cv2.VideoCapture('test4.mp4')
cap = cv2.VideoCapture(0)

while True:
    
    _, img = cap.read()
    
    #print(h,w)
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    #img = cv2.resize(img, (352,640))
    h, w, _ = img.shape
    boxes = []
    confidences = []
    class_ids = []
    
    for out in layerOutputs:
        for detection in out:
           scores = detection[5:]
           class_id = np.argmax(scores)
           confidence = float(scores[class_id])
           if confidence > 0.5:
               center_x = int(detection[0]*w)
               center_y = int(detection[1]*h)
               b_w = int(detection[2]*w)
               b_h = int(detection[3]*h)
               x = int(center_x - b_w/2)
               y = int(center_y - b_h/2)
               
               boxes.append([x,y,b_w,b_h])
               confidences.append(confidence)
               class_ids.append(class_id)
    
    print(len(boxes))
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)   #Gives indices of those boxes which are left out after eliminating redundent boxes
    try:
        print(indexes.flatten())
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        
        for i in indexes.flatten():
            x, y, b_w, b_h = boxes[i]
            label = str(classes[class_ids[i]])
            c = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+b_w, y+b_h), color, 2)
            cv2.putText(img, label + ' ' + c, (x, y+20), font, 2, (0,255,0), 2)
            print(x,y,b_w,b_h)
        
    except:
        pass
        
    out_vid.write(img)
    cv2.imshow('vid', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
out_vid.release()
cv2.destroyAllWindows()
