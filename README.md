# indoor_vision_navigation
It is a CNN approach to achieve indoor navigation.

## Training
Assuem the name of your dataset called "abc", you need to put all images into datasets/abc/temp. Then, create two empty directories named as train and test which contain an empty train.csv or test.csv respectively. Remember all the image names should such format, img_0001.jpg. The index must have length in 4.

There are several parameters you may need to change in shuffle_data.py which are lines 7-18. There are related to your datasets file path and data set distributions.

After all the parameters are ready, you may simply execute the following.

```
python3 shuffle_data.py
```

![image](https://github.com/vincent51689453/indoor_vision_navigation/blob/main/git_image/data_shuffle.png)

**However, you need to modify the csv file manually to label each image in the current stage. You may modify manual_label.py if it is useful for you.**

## Network Output

| Labels | Directions |
| ------ | ---------- |
| 0      | Forward    |
| 1      | Left       |
| 2      | Right      |
| 3      | Stop       |

## Tensorboard
```
tensorboard --logdir log/
```
You can access tensorboard by \<ip_address\>:6006 .