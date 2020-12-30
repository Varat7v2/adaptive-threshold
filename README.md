This project is based on the paper: [Adaptive Threshold for Better Performance of the Recognition and Re-identification Models](https://arxiv.org/abs/2012.14305)

This project can be tested on [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/), but because of the larger identities, its computationally time consuming. Hence, for testing purpose, I have prepared my own dataset of [top highly paid athletes listed on Forbes magazine (2018)](https://www.forbes.com/sites/kurtbadenhausen/2018/06/13/full-list-the-worlds-highest-paid-athletes-2018/?sh=47e592177d9f) using [online-dataset-generator](https://github.com/Varat7v2/online-dataset-maker). The Athletes dataset can be downloaded from [here](). Similarly, the model for face-detection and facenet can be downloaded from [here](https://drive.google.com/drive/folders/1okfoM_pxEUupdjYBzy7PyL0Y-0Swlym_?usp=sharing).

Clone the project
```
git clone https://github.com/Varat7v2/adaptive-threshold.git
```

Install necessary dependencies required for running this project.

```
pip install -r requirements.txt
```

We need to make necessary changes like dataset source path, models path etc., in a [config](https://github.com/Varat7v2/adaptive-threshold/blob/master/adaptive_config.py) file.

Finally, we can run the [main file](https://github.com/Varat7v2/adaptive-threshold/blob/master/adaptive-threshold-main.py) for adaptive threshold
```
python adaptive-threshold-main.py
```
