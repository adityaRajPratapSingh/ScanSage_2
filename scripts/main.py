from ultralytics import YOLO
import torch

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f'Using device: {device}')

#since the cuda versions of torch and tochvision are installed there is actually no need to specify device it will automatically choose the GPU its more important when there are multiple GPUs

model = YOLO("../ptWeights/yolov8n.pt")

# the __name__ stores the name of the current file that its in the code block that this creates gets executed only when this code is directly being run
#if this python code is beoing run as a script in some other module the __name__ var will have that modules name and this will not execute.
if __name__ == '__main__':
    model.train(data='config.yaml', epochs=25, device=0, imgsz=640, amp=False)
