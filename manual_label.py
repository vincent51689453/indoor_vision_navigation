import csv

with open('datasets/IC_Stairs/train/train.csv',mode='r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        index = int(row[0][4:8])
        if (((index>=318)and(index<=535))or\
        ((index>=638)and(index<=850))or\
        ((index>=926)and(index<=1127))or\
        ((index>=1217)and(index<=1354))):
            with open('train_new.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([row[0],2])

        elif((index>=1687)and(index<=1842)):
            with open('train_new.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([row[0],1])

        else:
             with open('train_new.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([row[0],0])
        
