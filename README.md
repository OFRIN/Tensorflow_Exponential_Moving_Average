# Exponential Moving Average (EMA)

## # Run
1. generate dataset
```sh
python ./dataset/Generate_Dataset.py
```

2. train (Base and EMA)
```sh
python Train_Base.py
python Train_with_EMA.py
```

3. run tensorboard
```sh
tensorboard --logdir logs
```

## # Result
- Loss is almost similar to Base and EMA.
- Accuracy is that EMA is more stable that Base. 

1. Base
![res](./res/base.PNG)

2. EMA
![res](./res/ema.PNG)

3. Base + EMA
![res](./res/combination.PNG)

## # Reference
- https://medium.com/datadriveninvestor/exponentially-weighted-average-for-deep-neural-networks-39873b8230e9

