from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


video_path_in = "D:\\Apps\\yolo3\\vids\\8_edited.mp4"
video_path_out = '{}_out.mp4'.format(video_path_in)

capture=cv2.VideoCapture(video_path_in)
assert capture.isOpened(), "error opening the video"
ret, frame=capture.read()
w, h, _=frame.shape
out=cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(capture.get(cv2.CAP_PROP_FPS)), (w,h))

model=YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")
threshold=0.2

region_points1 = [(400, 20), (400, 700)]
region_points2=[(800,20),(800,700)]
classes=[0,1]

counter1=object_counter.ObjectCounter()
counter1.set_args(view_img=True, view_in_counts=True, view_out_counts=False, reg_pts=region_points1, classes_names=model.names, draw_tracks=True, line_thickness=1)
counter2=object_counter.ObjectCounter()
counter2.set_args(view_img=True, view_in_counts=True, view_out_counts=True, reg_pts=region_points2, classes_names=model.names, draw_tracks=True, line_thickness=1)

while capture.isOpened():
    ret, frame=capture.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1920,1080))
    tracks=model.track(frame, persist=True, show=False, classes=classes)

    frame=counter1.start_counting(frame,tracks)
    counter2.start_counting(frame,tracks)
    out.write(frame)


print("counter1")
print(counter1.class_wise_count)
print("counter2")
print(counter2.class_wise_count)
capture.release()
out.release()
cv2.destroyAllWindows()