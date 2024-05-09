from ultralytics import YOLO
import os

model = YOLO("D:\\Apps\\yolo3\\ptWeights\\weights\\best.pt")

source = "D:\\Apps\\yolo3\\vids\\pic.png"

results = model(source)  # list of Results objects
