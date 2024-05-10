import os
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import threading
import time


def count_total():
    while True:
        time.sleep(5)
        total_humans_in, total_cards_in = 0, 0
        if counter.class_wise_count.get('human') and counter.class_wise_count.get('card'):
            total_humans_in = max(counter.class_wise_count['human']['IN'], counter.class_wise_count['human']['OUT'])
            total_cards_in = max(counter.class_wise_count['card']['IN'], counter.class_wise_count['card']['OUT'])
        elif counter.class_wise_count.get('human'):
            total_humans_in = max(counter.class_wise_count['human']['IN'], counter.class_wise_count['human']['OUT'])
            total_cards_in = 0
        elif counter.class_wise_count.get('card'):
            total_humans_in = 0
            total_cards_in = max(counter.class_wise_count['card']['IN'], counter.class_wise_count['card']['OUT'])
        print('TOTAL HUMANS AND TOTAL CARDS IN: ', total_humans_in, total_cards_in)


def get_overlap(box1, box2):
    """
    Implement the relative overlap between box1 and box2

    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box

    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)

    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)

    # compute the overlapped area w.r.t area of the smallest bounding box
    overlap = inter_area / min(box1_area, box2_area)
    return overlap


capture = cv2.VideoCapture("D:\\Apps\\yolo3\\vids\\9.mp4")

model = YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")

region_points1 = [(550, 538), (800, 538), (800, 38), (550, 38)]
classes = [0, 1]

counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, view_in_counts=True, view_out_counts=True, reg_pts=region_points1,
                 classes_names=model.names, draw_tracks=True, line_thickness=1)

t1 = threading.Thread(target=count_total)
t1.start()

already_detected = []

while True:

    counter_for_name=0

    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 576))
    tracks = model.track(frame, persist=True, show=False, classes=classes)
    counter.start_counting(frame, tracks)

    new_detections = []
    results = model(frame)[0]
    for result in results:
        h_x1, h_y1, h_x2, h_y2 = -1, -1, -1, -1
        c_x1, c_y1, c_x2, c_y2 = -1, -1, -1, -1
        if result.boxes.xyxy.tolist() not in already_detected:
            new_detections.append(result.boxes.xyxy.tolist())
            if result.boxes.cls == 0:
                h_x1, h_y1, h_x2, h_y2 = result.boxes.xyxy[0]
            if result.boxes.cls == 1:
                c_x1, c_y1, c_x2, c_y2 = result.boxes.xyxy[0]
        if get_overlap([h_y1, h_x1, h_y2, h_x2], [c_y1, c_x1, c_y2, c_x2]) < 0.9:
            screenshot = frame.copy()
            filename = f'screenshot{counter_for_name}.jpg'
            cv2.imwrite(filename, screenshot)
            print("SCREENSHOT CAPTURED")
        already_detected.extend(new_detections)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

t1.join()
capture.release()
cv2.destroyAllWindows()
