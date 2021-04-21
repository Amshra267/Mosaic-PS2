import cv2
import numpy as np
import math

def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    classes = [2,5,7]
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            #print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id in classes:
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    img = cv2.resize(img, (640, 480))
    cv2.imshow("Image", img)

def image_detect(img): 
    detect_l = 0
    plate_detector = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bxs = plate_detector.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=6)
    if len(bxs)==1:
        detect_l = 1
        for bx in bxs:
            cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[0]+bx[2]), int(bx[1]+bx[3])), (0, 0, 255), 2)

    return img, detect_l

def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    tracker = cv2.TrackerKCF_create()
    images = []
    cap = cv2.VideoCapture(video_path)
    start_pos = 0
    curr_pos = 0
    frame_count = 0
    d = 0
    num_per_count = 0
    init = False
    

    while True:
        if init:
            _, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (416, 416))
            #print(frame.shape)
            success, target_box = tracker.update(frame)
            frame_count += 1
            num_per_count += 1
            #print(success)
            if success:
                #print("SuCCEss")
                (x, y, w, h) = [int(v) for v in target_box]
                print("Centre coordinates: ", (x+w/2), (y+h/2))
                curr_pos = [x+w/2, y+h/2]
                d += math.dist(start_pos, curr_pos)
                speed = d/frame_count
                print("Distance Travelled: ", d)
                print("Speed: ", speed)
                print("Frame count: ", frame_count)
                # if speed>3:
                if num_per_count<3:
                    images.append(frame[y:y+h, x:x+w])
                start_pos = [x+w/2, y+h/2]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("initED, nOt succESS")
                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
                indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
                #print(np.array(boxes).shape)
                boxes = np.array(boxes)[indexes]
                frame_count = 0
                num_per_count = 0
                d = 0
                #print(boxes)
                try:
                    boxes = sorted(boxes, key= lambda x: (x[0][2]*x[0][3]), reverse = True)
                    if boxes is not None:
                        target_box = boxes[0][0]
                        #print(target_box)
                        (x, y, w, h) = [int(v) for v in target_box]
                        start_pos = [x+w/2, y+h/2]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        tracker = cv2.TrackerKCF_create()
                        #print(tracker)
                        success = tracker.init(frame, (x, y, w, h))
                        #print(success)
                except:
                    boxes = sorted(boxes, key= lambda x: (x[2]*x[3]), reverse = True)
                    if boxes is not None:
                        target_box = boxes[0]
                        #print(target_box)
                        (x, y, w, h) = [int(v) for v in target_box]
                        start_pos = [x+w/2, y+h/2]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        tracker = cv2.TrackerKCF_create()
                        #print(tracker)
                        success = tracker.init(frame, (x, y, w, h))
                        #print(success)
                
        else:
            #print("iNIt")
            init = True
            _, frame = cap.read()
            frame = cv2.resize(frame, (416, 416))
            height, width, channels = frame.shape
            #print(frame.shape)
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            boxes = np.array(boxes)[indexes]
            boxes = np.clip(boxes, 0, 416)
            # print(boxes)
            boxes = sorted(boxes, key= lambda x: (x[0][2]*x[0][3]), reverse = True)
            if boxes is not None:
                target_box = (boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3])
                start_pos = [target_box[0]+target_box[2]/2, target_box[1]+target_box[3]/2]
                frame_count = 0
                num_per_count = 0
                #print(target_box)
                success = tracker.init(frame, target_box)
                #print(success)
        if frame is not None:
            cv2.imshow("Image", frame)
        #draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    return images



if __name__ == "__main__":
    video_path = input("Provide video path: \n")
    images = start_video(video_path)

    for idx, img in enumerate(images):
        img = cv2.resize(img, (160, 120))
        img, l = image_detect(img)
        if l == 1:
            cv2.imshow(str(idx), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()