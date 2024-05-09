from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


video_path_in = "D:\\Apps\\yolo3\\vids\\11.mp4"
video_path_out = '{}_out.mp4'.format(video_path_in)

capture=cv2.VideoCapture(video_path_in)
assert capture.isOpened(), "error opening the video"
ret, frame=capture.read()
w, h, _=frame.shape
out=cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(capture.get(cv2.CAP_PROP_FPS)), (w,h))

model=YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")
threshold=0.2

region_points = [(40, 400), (1240, 400)]
classes=[0,1]

counter=object_counter.ObjectCounter()
counter.set_args(view_img=True, view_in_counts=True, view_out_counts=False, reg_pts=region_points, classes_names=model.names, draw_tracks=True, line_thickness=1)

while capture.isOpened():
    ret, frame=capture.read()
    if not ret:
        break
    tracks=model.track(frame, persist=True, show=False, classes=classes)

    frame=counter.start_counting(frame,tracks)
    out.write(frame)

#print(counter.in_counts)
capture.release()
out.release()
cv2.destroyAllWindows()