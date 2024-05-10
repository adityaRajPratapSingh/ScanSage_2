from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import supervision as sv
import numpy as np


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


video_path_in = "D:\\Apps\\yolo3\\vids\\8_edited.mp4"
video_path_out = '{}_out.mp4'.format(video_path_in)



capture = cv2.VideoCapture(video_path_in)
assert capture.isOpened(), "error opening the video"
ret, frame = capture.read()
w, h, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(capture.get(cv2.CAP_PROP_FPS)), (w, h))



model = YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")



region_points1 = [(550,538), (800,538), (800,38), (550,38)]
classes = [0, 1]



counter1 = object_counter.ObjectCounter()
counter1.set_args(view_img=True, view_in_counts=True, view_out_counts=True, reg_pts=region_points1,
                  classes_names=model.names, draw_tracks=True, line_thickness=1)


previous_Detections=[]

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    results = model(frame)[0]

    #detections=results.boxes[0].cpu().numpy()
    detections=results.boxes

    class1_detections = detections[detections[:, 5] == 0]
    class2_detections = detections[detections[:, 5] == 1]

    for class1_det in class1_detections:
        for class2_det in class2_detections:
            if get_overlap(class1_det[:4], class2_det[:4])<0.9:
                screenshot = frame.copy()
                cv2.imwrite('screenshot.jpg', screenshot)
                print("SCREENSHOT CAPTURED")
                break

    '''c_x1,c_y1,c_x2,c_y2=0,0,0,0
    h_x1,h_y1,h_x2,h_y2=0,0,0,0'''
    '''results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.5, class_agnostic=False)
    detections_class_0 = detections[detections.class_id == 0] #human
    detections_class_1 = detections[detections.class_id == 1] #card

    Ious = sv.Detections.utils.box_iou_batch(detections_class_0, detections_class_1)
    count=0
    while Ious:
        if Ious[count][count]!=1
            
        count+=1'''
    '''new_Detections=[]

    while detections:
        if detections not in previous_Detections:
            new_Detections.append(detections)
            if detections.class_id==0:
                h_x1, h_y1, h_x2, h_y2 = detections.xyxy[0]
                """the detections cycy returns a mumpy array that is of the type [n,4] meaning that it has n rows and each of the rows has 4 colums and wach of those columns contain the 4 coordinate values
                for each bounding box so if i say detections.xyxy.tolist() then each of thta list item will contain 4 coordinate values"""
            if detections.class_id==1:
                c_x1,c_y1,c_x2,c_y2=detections.xyxy[0]
            if not (h_x1<c_x1<h_x2 and h_x1<c_x2<h_x2 and h_y2<c_y1<h_y1 and h_y2<c_y2<h_y1):
                screenshot = frame.copy()
                cv2.imwrite('screenshot.jpg', screenshot)
                print("SCREENSHOT CAPTURED")
                break

    previous_Detections.extend(new_Detections)'''

    frame = cv2.resize(frame, (1024, 576))
    tracks = model.track(frame, persist=True, show=False, classes=classes)
    frame = counter1.start_counting(frame, tracks)
    out.write(frame)


total_humans_in, total_cards_in = 0, 0
if counter1.class_wise_count.get('human') and counter1.class_wise_count.get('card'):
    total_humans_in = max(counter1.class_wise_count['human']['IN'],counter1.class_wise_count['human']['OUT'])
    total_cards_in = max(counter1.class_wise_count['card']['IN'],counter1.class_wise_count['card']['OUT'])
elif counter1.class_wise_count.get('human'):
    total_humans_in = max(counter1.class_wise_count['human']['IN'],counter1.class_wise_count['human']['OUT'])
    total_cards_in=0
else:
    total_humans_in=0
    total_cards_in = max(counter1.class_wise_count['card']['IN'],counter1.class_wise_count['card']['OUT'])
print('total humans and total cards in: ', total_humans_in, total_cards_in)


capture.release()
out.release()
cv2.destroyAllWindows()
