## Back-propagation Neural Network
Command line instruction:
we recommend using python3 to run this program
The program includes two .py files: preProcessing.py and NeNet.py.

### Pre-Process
the input parameters to the preProcessing.py file are:
+ complete input path of the raw dataset, or instead we stored the url of the three following dataset, you can put the name for that particular dataset:
  - ds1: https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
  - ds2: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
  - ds3: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
+ complete output path of the pre-processed dataset. For example 'postProcessed.csv' can be a path for the output file.

for example **" python3 preProcessing.py ds1 'postProcessed.csv' "** 
The above would imply that the training dataset is 'ds1' which is the first dataset listed above. The output path is 'currentDirectory/postProcessed.csv'

### Back-propagation
The input parameters to the NeNet.py
are as follows:
+ input dataset – a complete path the post-processed input dataset which you specfied for the output path of the preProcessing.py

+ training percent – percentage of the dataset to be used for training
+ maximum_iterations – Maximum number of iterations that your algorithm will run. This
parameter is used so that your program terminates in a reasonable time.
+ number of hidden layers
+ number of neurons in each hidden layer


for example **" python3 NeNet.py 'postProcessed.csv' 80 4000 2 10 10 "**
The above would imply that the dataset is 'postProcessed.csv', the percent of the dataset to be used for
training is 80%, the maximum number of iterations is 200, and there are 2 hidden layers with
(4, 2) neurons. Your program would have to initialize the weights randomly
