# Datasets file path
dataset_name = 'IC_Stairs'
testing_csv_path = './datasets/'+ dataset_name +'/test/test.csv'
testing_img_path = './datasets/'+ dataset_name +'/test/'
training_csv_path = './datasets/'+ dataset_name +'/train/train.csv'
training_img_path = './datasets/'+ dataset_name +'/train/'

# Hyperparameters
param_batch_size = 64
param_learning_rate = 0.001
num_epoches = 10

# Network parameters
input_channel = 3   # RGB Image 
output_channel = 4  # Number of categories