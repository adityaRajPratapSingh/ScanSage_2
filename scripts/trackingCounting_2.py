from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import supervision as sv



video_path_in = "D:\\Apps\\yolo3\\vids\\7_edited.mp4"
video_path_out = '{}_out.mp4'.format(video_path_in)



capture = cv2.VideoCapture(video_path_in)
assert capture.isOpened(), "error opening the video"
ret, frame = capture.read()
w, h, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(capture.get(cv2.CAP_PROP_FPS)), (w, h))



model = YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")



#region_points1 = [(1600,1060), (1900,1060), (1900,20), (1600,20)]
region_points1 = [(550,538), (800,538), (800,38), (550,38)]
classes = [0, 1]



counter1 = object_counter.ObjectCounter()
counter1.set_args(view_img=True, view_in_counts=True, view_out_counts=True, reg_pts=region_points1,
                  classes_names=model.names, draw_tracks=True, line_thickness=1)


while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
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
