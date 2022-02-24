# AILN
#The code for paper 'Adaptive Intention Learning for Session-based Recommendation'.
## Requirements
- `Anaconda=4.8.5`
- `python=3.6`
- `pytorch==1.1.0`
- `CUDA==10.1`
- `cuDNN==7.5.1`


## Usage
1. Install required packages.
2. run <code>python main_tmall.py</code> to train and evaluate AILN with Tmall dataset. Similar scripts are prepared for Tafeng (<code>python main_tafeng.py</code>),Yoochoose(<code>python main_yoo.py</code>) 
## Datasets
- The datasets used in our paper are organized in [IJCAI/],[Tafeng/] and [Yoo/],  where each data file contains the instances of training, validation and test sets.
Tmall is collected from https://tianchi.aliyun.com/dataset/dataDetail?dataId=42,
Tafeng is from https://www.kaggle.com/chiranjivdas09/ta-feng-grocerydata ,
Yoochoose is from https://2015.recsyschallenge.com/challenge.html.
