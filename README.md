# Machine Learing methods for Surgical Assessment
Respository contains some machine learning models developed for classifying surgical skills (Novice, Intermediate and Expert) or comparing surgical skills (Similarity value) from surgical performance datasets.

## Datasets
The scripts are written for a dataset generated through our APSS (Autonomous Platform for Surgical Simualtion) simulator. The data generated is not uploaded here, since its still under wraps and being worked on. The dataset contains kinematics for two surgical procedures: Pericardiocentesis (PCS) and Thoracentesis TCS). PCS has 2 surgical actions and TCS has 4. So the scripts are written to train a total of 6 separate models using the same network, to classify skills for each surgical task. 

There is a publically available dataset called JIGSAWS which contains surgical performance data in the form of kinematics and videos, that is made available here. The dataset contains 3 surgical tasks performed using the surgical robot called DaVinci. There are differences in features and labels between our dataset and JIGSAWS, so naturally the scripts need to be editted to train for this dataset, however the overall fucntion remains the similar. 

## Data Visualization
The script was writtent to observe the distribution of features in the dataset. Mainly to recognize the outliers and determine the thresholds (min, max) for normalization of the features. 

## Feature Engineering
The process entails the following:
1. The dataset was first manually cleaned to remove the parts of actions not related to surgical tasks.
2. Feature transformaion for orientation values to convert Euler Angles to Quaternions.
3. Additional features such as linear and rotational velocities were calculated through available time stamps for position and orientaiton values.
4. Normalization of data for training.

## Training and Testing
Currently contains scripts for 
1. 1D-CNN
2. LSTM
3. Siamese Convolution Neural Network
Note: All the scirpts perform data augmentation using sliding window method. 

