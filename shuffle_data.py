import random
import os
import csv
import time

# Datasets parameters
total_samples = 2378
training_portion = int(0.7*total_samples)
testing_portion = total_samples - training_portion

sample_record = []

# Creating testing and training samples
dataset_name = 'datasets/IC_Stairs/temp'
train_dir = 'datasets/IC_Stairs/train'
test_dir = 'datasets/IC_Stairs/test'
train_csv = 'datasets/IC_Stairs/train/train.csv'
test_csv = 'datasets/IC_Stairs/test/test.csv'

def index_padding(num):
    num = str(num)
    L = len(num)
    if(L == 4):
        return num
    elif(L == 3):
        return '0'+num
    elif(L == 2):
        return '00'+num
    elif(L == 1):
        return '000' + num
    else:
        return '0000'

# Initialization
for x in range(0,total_samples):
    # 1 stands for pending to process
    # 0 means finished 
    sample_record.append(1)


# Move images to training
i = 0
while(i<training_portion):
    index = random.randint(0,total_samples-1)
    if(sample_record[index]==1):
        # Generate file names
        file_index = index_padding(index)
        file_name = 'img_' + file_index +'.jpg'
        cmd = 'mv datasets/IC_Stairs/temp/img_'+file_index+'.jpg ' + train_dir
        # Move samples
        print("Execute: ",cmd)
        os.system(cmd)
        # Append csv
        with open(train_csv, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([file_name,0])

        sample_record[index] = 0
        i += 1

print("Training data are ready...")
time.sleep(3)

j = 0
k = 0
# Move images to testing
while(j<total_samples):
    if(sample_record[j]==1):
        # Generate file names
        file_index = index_padding(j)
        file_name = 'img_' + file_index +'.jpg'
        cmd = 'mv datasets/IC_Stairs/temp/img_'+file_index+'.jpg ' + test_dir
        # Move samples
        print("Execute: ",cmd)
        os.system(cmd)
        # Append csv
        with open(test_csv, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([file_name,0])
        k+=1

    j += 1 


# Output training directory report
print("Training Dataset ok!")
percentage = round((i/total_samples*100),2)
print("Number of training samples moved:{} ={}%".format(i,percentage))

# Output testing directory report
print("Testing Dataset ok!")
percentage = round((k/total_samples*100),2)
print("Number of training samples moved:{} ={}%".format(k,percentage))
print("Total Samples:",total_samples)