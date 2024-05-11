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
def bbox_iou(bbox1, bbox2):

  area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
  area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

  x_min = max(bbox1[0], bbox2[0])
  y_min = max(bbox1[1], bbox2[1])
  x_max = min(bbox1[2], bbox2[2])
  y_max = min(bbox1[3], bbox2[3])

  intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

  iou = intersection_area / (area1 + area2 - intersection_area)

  return iou

capture = cv2.VideoCapture("D:\\Apps\\yolo3\\vids\\8_edited.mp4")
model = YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")
region_points1 = [(550, 538), (800, 538), (800, 38), (550, 38)]
classes = [0, 1]
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, view_in_counts=True, view_out_counts=True, reg_pts=region_points1, classes_names=model.names, draw_tracks=True, line_thickness=1)
t1 = threading.Thread(target=count_total)
t1.start()
tracked_humans={}
tracked_cards={}
counter_for_name=0
while True:

    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 576))
    tracks = model.track(frame, persist=True, show=False, classes=classes)
    counter.start_counting(frame, tracks)

    '''for track in tracks:
        track_id = track.id
        track_class = track.cls
        track_box = track.xyxy[0]

        if track_class==0:
            tracked_humans[track_id]=track_box
        elif track_class==1:
            tracked_cards[track_id]=track_box'''
    for results in tracks:
        for track in results.xyxy:
            track_id = track[0]  # Accessing the first element of the bounding box
            track_class = int(track[5])  # Accessing the class ID
            track_box = track[1:5]  # Accessing the bounding box coordinates

            if track_class == 0:
                tracked_humans[track_id] = track_box
            elif track_class == 1:
                tracked_cards[track_id] = track_box

    for human_id, human_box in tracked_humans.items():
        no_overlap = True  # Flag to indicate no overlap found

        for card_id, card_box in tracked_cards.items():
            if bbox_iou(human_box, card_box) > 0:  # Any overlap detected
                no_overlap = False
                break
        if no_overlap:
            screenshot = frame.copy()
            filename = f'screenshot{counter_for_name}.jpg'
            cv2.imwrite(filename, screenshot)
            print("SCREENSHOT CAPTURED")
            counter_for_name += 1

t1.join()
capture.release()
cv2.destroyAllWindows()
