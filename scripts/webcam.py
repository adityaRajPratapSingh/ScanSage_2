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
        else:
            total_humans_in = 0
            total_cards_in = max(counter.class_wise_count['card']['IN'], counter.class_wise_count['card']['OUT'])
        print('TOTAL HUMANS AND TOTAL CARDS IN: ', total_humans_in, total_cards_in)

capture = cv2.VideoCapture(0)

model = YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")

region_points1 = [(550, 538), (800, 538), (800, 38), (550, 38)]
classes = [0, 1]

counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, view_in_counts=True, view_out_counts=True, reg_pts=region_points1,
                 classes_names=model.names, draw_tracks=True, line_thickness=1)

t1 = threading.Thread(target=count_total)
t1.start()
while True:
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 576))
    tracks = model.track(frame, persist=True, show=False, classes=classes)
    counter.start_counting(frame, tracks)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

t1.join()
capture.release()
cv2.destroyAllWindows()
