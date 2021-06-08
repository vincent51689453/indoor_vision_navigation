import cv2

img = cv2.imread('./datasets/gazebo_ic382/frame_buffer/frame_0.jpg')
print(img)
cv2.imshow('test',img)
cv2.waitKey(0)