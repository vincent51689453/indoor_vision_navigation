import cv2
import torch
from torch import nn
from torchvision import transforms
from torchvision.io import read_image
from torch.autograd import Variable

# Video source
video_input_path = 'datasets/IC_Stairs/ic_stairs.mp4'
capture = cv2.VideoCapture(video_input_path)

# Load network with CUDA support
network_save_path = 'models/navigation_net.pt'
navigation_CNN = torch.load(network_save_path)
print(navigation_CNN)
if torch.cuda.is_available():       
    navigation_CNN = navigation_CNN.cuda()  

transformer = transforms.ToTensor()

while (capture.isOpened()):
    # Load video frames
    ret,frame = capture.read()

    # Resize frame
    frame = cv2.resize(frame,(720,480))

    # np array to tensor
    # # Expand dimensions from [3,480,720] to [1,3,480,720]
    image_tensor = transformer(frame)   
    image_tensor = torch.unsqueeze(image_tensor,0)
    if torch.cuda.is_available():
            image_tensor = Variable(image_tensor).cuda()

    # Inference and search for max probability
    direction = navigation_CNN(image_tensor)
    _,predicted = torch.max(direction.data,1)
    direction_label = predicted.item()

    # Draw Pointer
    # Label 0: Forward
    if(direction_label == 0):
        start_pt_x = int(720/2)
        start_pt_y = int(480/2 + 60)
        end_pt_x = start_pt_x
        end_pt_y = int(480/2 - 60)
        frame = cv2.arrowedLine(frame,(start_pt_x,start_pt_y),(end_pt_x,end_pt_y),(255,0,255),5)
    # Label 1: Left
    if(direction_label == 1):
        start_pt_x = int(720/2-60)
        start_pt_y = int(480/2)
        end_pt_x = int(720/2+60)
        end_pt_y = start_pt_y
        frame = cv2.arrowedLine(frame,(start_pt_x,start_pt_y),(end_pt_x,end_pt_y),(255,0,255),5)

    # Display
    cv2.imshow('Navigation',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()