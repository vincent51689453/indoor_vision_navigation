# indoor_vision_navigation
It is a CNN approach to achieve indoor navigation.

## Training
Assuem the name of your dataset called "abc", you need to put all images into datasets/abc/temp. Then, create two empty directories named as train and test which contain an empty train.csv or test.csv respectively. Remember all the image names should such format, img_0001.jpg. The index must have length in 4.

There are several parameters you may need to change in shuffle_data.py which are lines 7-18. There are related to your datasets file path and data set distributions.

After all the parameters are ready, you may simply execute the following to prepare the dataset.

```
python3 shuffle_data.py
```

![image](https://github.com/vincent51689453/indoor_vision_navigation/blob/main/git_image/data_shuffle.png)

if you need to resize all the image in the dataset, you can run and edit
```
python3 manual_resize.py
```

**However, you need to modify the csv file manually to label each image in the current stage. You may modify manual_label.py if it is useful for you.**

When the dataset is ready, you can execute the following to start training.
```
python3 train.py
```

## Tensorboard
```
tensorboard --logdir log/
```
You can access tensorboard by \<ip_address\>:6006 .

![image](https://github.com/vincent51689453/indoor_vision_navigation/blob/main/git_image/tensorboard.png)

## Network Output

| Labels | Directions |
| ------ | ---------- |
| 0      | Forward    |
| 1      | Left       |
| 2      | Right      |
| 3      | Stop       |

## Network deployment
After modifying the input source of inference.py and network path, you can run
```
python3 inference.py
```
![image](https://github.com/vincent51689453/indoor_vision_navigation/blob/main/git_image/navigation_net_v1.gif)