
# Documentation 

This documentation is used for Accident_predict_rnn.py, of which the tensorflow is imported to run RNN algorithm in order to predict accidents. For more information please see [this].



**Environmental requirements** : The following platform is currently where I run my code

- **Python** - The used Python version is Python 2.7.12.
- **Tensorflow** -  Version r.12
- **OS** - Mac OS X EI Captitan

----------
[TOC]

## Functions

> In this section, mainly explain the functions I defined in the source code file.
### convert_data
By this function, we can convert the original given data to required format we use in the RNN model. Some data we human beings can understand but the computer does not know what it means. It is necessary to convert the original data as a way the software can understand

There are three parts:
- Convert the accident type
``` 
ac_types = {
        "NONE": "0",
        "A1": "1",
        "A2": "2",
        "A3": "3" }
```
- Convert the weekend 
Express the date  to  a boolean value. From Monday to Friday, it is workday. Saturday and Sunday we treat as weekend.
``` 
if ac_time.weekday() >= 5:
                strWeek = "1,0"
            else:
                strWeek = "0,1"
```
- Convert time
We divide 24 hours a day into four different stages. Then use boolean value to express the specific time stage.
``` 
if ac_time.hour <= 6 or ac_time.hour >= 22:
                strTime = "1,0,0,0"
            elif ac_time.hour >= 6 and ac_time.hour <= 10:
                strTime = "0,1,0,0"
            elif ac_time.hour >= 10 and ac_time.hour <= 16:
                strTime = "0,0,1,0"
            elif ac_time.hour >= 16 and ac_time.hour <= 22:
                strTime = "0,0,0,1"
```

After we convert all these data, we write these data to a new file. As shown in the function 
```
def convert_data(srcFile, dstFile)
```
*srcFile*  is the original source file , *dstFile* is the new file.
### normalize
Some original data are not good for use because of their different scale. This function will normalize the data between 0 and 1. 
```
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))
```
### data_import
When we finishing convert the data and normalize data, we will import the data preparing for the RNN model.

```
def data_import(file, delimiter=',', cols=(), normalize_cols=()):
```
Argument *file* is the targeted data file we used in the RNN model, *cols* is the columns we choose  from the targeted file as input for RNN model. *normalize_cols* is the columns to be normalized.


### train
This function is mainly used for training the data and getting the optimal parameters, which is prepared for the test in the next step.
```
def train(train_x, train_y, learning_rate=0.02, epochs=1, batch_n=1, input_n=1, hidden_n=4)
```
- *train_x* is the input training data set
- *train_y* is the desired output in the training data set
- *learning_rate* is the learning rate in the BPTT algorithm , default value =0.02
- *epochs* is epoch size, default value = 1
- *batch_n* is batch size, default value = 1
- *input_n* is the input vector dimension, default value = 1.
- *hidden_n* is the number of neurons in the hidden layer, default value = 4

### predict

After training the data, we have the optimal coefficients for the RNN model, then apply this model to predict accident by using test data set.
```
def predict(test_x, test_y, batch_n)
```
- *test_x* is the input for test.
- *test_y* is the actual output in the test data, in order to compare with prediction of the model.
- *batch_n* is the batch size.


### print_scores
This function is used to calculate the objective scores. Then print the result as required in a easy way.
```
 def print_scores():
        ScoreAccidents = float(self.predict_acc) / self.total_acc
        ScoreAccidentTFrames = float(self.predict_acc_tframes) / self.total_acc_tframes
        ScoreNonAccidents = float(self.predict_no_acc_tframes) / self.total_no_acc_tframes
             
        print("ScoreAccidents:{%f}" % ScoreAccidents)
        print("ScoreNonAccidents:{%f}" % (ScoreNonAccidents))
        print("ScoreAccidents Time Frames:{%f}" % ScoreAccidentTFrames)
        print("Score1:{%f}" % ((ScoreAccidents + ScoreNonAccidents) / 2))
        print("Score2:{%f}" % (float(self.predict_acc) / self.total_no_acc_tframes))
```

## Configuration

There is a dictionary used for configuration. This is configuration is necessary to train your model.
```
config = {
        "batch_n": 1,
        "epochs": 4,
        "train_start": 0,
        "train_end": 4000,
        "cols": (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        "normalize_cols": (3, 4, 5, 6),
    }
```
- *batch_n* is the batch size 
- *epochs* is the epoch size
- *train_start* is the start line in the training data set
-  *train_end* is the end line in the training data set
-  *cols* is the columns you choose for training model in the data set
-  *normalize_cols* is the columns you choose to normalize in the data set

## Feedback & Bug Report

Jack Zheng: *zzhengfei01@gmail.com*

        
----------
Thank you for reading this documentation. 


  [this]: https://github.com/zzf-technion/RNN_accident_forecast
