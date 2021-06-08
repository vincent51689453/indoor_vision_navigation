import cv2
import csv
"""
This scipts helps to resize all the images from test_original or train_original to test and train.
"""

with open('datasets/IC_Stairs/test/test.csv',mode='r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        name = row[0]
        complete_name = 'datasets/IC_Stairs/test_original/'+name
        img = cv2.imread(complete_name)
        img = cv2.resize(img,(720,480))
        save_path = 'datasets/IC_Stairs/test/'+name
        cv2.imwrite(save_path,img)
        cv2.imshow("buffer",img)
        cv2.waitKey(1)
print("Done")

