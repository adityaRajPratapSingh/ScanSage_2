from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import threading
import time
from datetime import datetime
from pymongo import MongoClient
import os

counter_for_name = 0


def upload_data(is_person_wearing_card: bool, image_path: str):
    # Connect to MongoDB
    client = MongoClient(
        'mongodb+srv://username:Singhabhii@cluster0.vgchusl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    )
    db = client['IdAnalyzer']
    collection_name = 'IdAnalyzer'

    date_obj = datetime.now()
    isCard = is_person_wearing_card
    isOnlyPerson = not is_person_wearing_card
    owner_id = ""

    # Read image file into binary data
    with open(image_path, 'rb') as f:
        image_data = f.read()

    document = {
        "date": date_obj,
        "isCard": isCard,
        "isPersonWithoutId": isOnlyPerson,
        "owner_id": owner_id,
        "imageData": image_data
    }

    result = db[collection_name].insert_one(document)
    print("DOCUMENT INSERTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def count_total():
    while True:
        time.sleep(5)
        wearing_card = bool
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
        print('------------------------------------------------------------------------------TOTAL HUMANS AND TOTAL CARDS IN: ', total_humans_in, total_cards_in)
        if total_humans_in == total_cards_in:
            wearing_card = True;
            screenshot = frame.copy()
            cv2.imwrite('screenshot.jpg', screenshot)
            path = "D:\\Apps\\yolo3\\scripts\\screenshot.jpg"
            upload_data(wearing_card, path)
            os.remove(path)

        else:
            wearing_card = False
            screenshot = frame.copy()
            cv2.imwrite('screenshot.jpg', screenshot)
            path = "D:\\Apps\\yolo3\\scripts\\screenshot.jpg"
            upload_data(wearing_card, path)
            os.remove(path)


capture = cv2.VideoCapture("D:\\Apps\\yolo3\\vids\\8_edited.mp4")
ret, frame = capture.read()

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

t1.join()
capture.release()
cv2.destroyAllWindows()
