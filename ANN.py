# Import python packages.
import numpy as np
from   numpy.random import seed
from   tensorflow import set_random_seed
from   scipy import integrate

import pandas as pd
from sklearn.preprocessing import StandardScaler

from   keras.utils import to_categorical
from   keras.models import Sequential
from   keras.layers import Dense, Dropout
from   keras.constraints import maxnorm
from   keras.optimizers import Adam

from   sklearn.model_selection import train_test_split

import pprint
from   openpyxl import Workbook
import os
import seaborn as sn
import matplotlib.pyplot as plt
Print = pprint.PrettyPrinter(indent = 4)


def differential_model(x, t, u, theta, equations):
    """
    General model in the form of a system of ODEs. Structured to be called by the scipy odeint package.

    Inputs
    x: array of state values
    t: time value
    u: array of time-invariant inputs
    theta: array of model parameters,
    equations: string-like object containing the expression of the first order derivatives
        

    Output: Array of dim(x) with the numerical values for the first order derivatives at the conditions specified by the inputs     

    """

    return eval(equations)


def measurement(x0, U, t, sigma, theta, equations): 
    '''
    Generates a sample in-silico by integrating the system of ODEs defined by the "equations" input

    Inputs:
        x0: array of initial states
        U: array of time-invariant system inputs
        t: scalar sampling time
        sigma: array of standard deviations for the uncorrelated measurement noise of the measured quantities involved in a sample [s_rel; s_const]
        theta: array of parameters used to integrate the assumed system model
        equations: string-like object containing the expression of the first order derivatives              

    Output:
        measurements

    '''

    X = integrate.odeint(differential_model, x0, [0.0, t], (U, theta, equations))
    std = np.sqrt(sigma[0] ** 2 / 100 * X[1] + sigma[1] ** 2)
    measurement = np.abs(X[1] + np.random.normal(0.0, std))
    return measurement
    

equation_codes = [
    "np.inner(np.array([[-1,0],[1,-1],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]    , theta[2]*np.exp(-theta[5]/8.314/u)*x[1]    ]))",
    "np.inner(np.array([[-1,0],[1,-1],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]    , theta[2]*np.exp(-theta[5]/8.314/u)*x[1]**2 ]))",
    "np.inner(np.array([[-1,0],[1,-1],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]**2 , theta[2]*np.exp(-theta[5]/8.314/u)*x[1]    ]))",
    "np.inner(np.array([[-1,0],[1,-1],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]**2 , theta[2]*np.exp(-theta[5]/8.314/u)*x[1]**2 ]))",
    "np.inner(np.array([[-1,-1],[1,0],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]    , theta[1]*np.exp(-theta[4]/8.314/u)*x[0]    ]))",
    "np.inner(np.array([[-1,-1],[1,0],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]    , theta[1]*np.exp(-theta[4]/8.314/u)*x[0]**2 ]))",
    "np.inner(np.array([[-1,-1],[1,0],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]**2 , theta[1]*np.exp(-theta[4]/8.314/u)*x[0]    ]))",
    "np.inner(np.array([[-1,-1],[1,0],[0,1]]), np.array([ theta[0]*np.exp(-theta[3]/8.314/u)*x[0]**2 , theta[1]*np.exp(-theta[4]/8.314/u)*x[0]**2 ]))"]

models = [compile(model, "<string>", "eval") for model in equation_codes]            
                                    

'''SET the right path for the file with model parameters'''
parameters_df = pd.read_csv('C:/Users/enry_/OneDrive/UNI/Python/Data/ModelParameters.csv')

parameters_df.label = parameters_df.label.astype(int)
labels_set = parameters_df['label'].values
Y = labels_set
parameters_set = [parameters_df.loc[n].values[2:] for n in range(len(parameters_df.index))]


# fixed experimental conditions
    # initial concentration of A B and C
x0 = np.array([ [249, 0, 0], [144, 0 , 0], [88, 0 , 0], [180, 0, 0],  [48, 0, 0],   ]) 
 
    # temperatures at which the experiments are conducted                                                 
u = np.array([  680, 599, 705, 640, 710,  ])   
                                                  
    # sampling times for each experiment
t = np.array([ 100, 200, 350 ])                    

    # noise on the concentration measurements
sigma = np.array([0.00,1.00])                                                  

np.random.seed(4)
                                            


# generation of the data to feed the network at fixed experimental conditions 
n_exp = int(len(u))
n_samples = int(len(t))

X = np.array([np.concatenate([ np.concatenate([
    
                measurement(x0[i], u[i], t[j], sigma, parameter, models[label - 1])
            
                            for j in range(n_samples) ])                      
                for i in range(n_exp) ])
              
        for label, parameter in zip(labels_set, parameters_set) ])
         


# 1.3) The size of the dataset, also represents the visible structure of the neural network;
'''CHANGE Input Size HERE'''
'''This part is need to adjust when the data is changing'''
X_size = np.shape(X)[1]                                                         # The input size of the neural network.
Y_size = 8                                                                      # The output size of the neural network.

# Section 2: Preprocessing the data. 
# 2.1) Fix random seed for reproducibility;
seed_num = 8
seed(seed_num)                                                                  # Random seed for result reproducibility.
set_random_seed(seed_num)

# 2.2) Spliting data into 'seen' (training & validation) and 'unseen' (testing);
X_seen, X_unseen, Y_seen, Y_unseen = train_test_split(X, Y, test_size = 0.2, 
                                                        random_state=seed_num)  # Spliting X & Y data into seen and unseen datasets.
                                                        
# 2.3) Scaling for the X dataset;
scaler = StandardScaler()                                                       # Set the scaling feature - removing the mean and scaling to unit variance.
scaler.fit(X_seen , X_unseen)                                                   # Set the scaling tool only on X data only.
x_seen = scaler.transform(X_seen)                                               # Transform the seen (training & validation) data using the scaler. 
x_unseen = scaler.transform(X_unseen)                                           # Transform the unseen (testing) data using the scaler.

# 2.4) Convert class vectors to binary class matrices (One-Hot Encoding) for Y dataset;
y_seen = to_categorical(Y_seen, Y_size + 1)                                     # One-Hot Encoding on Y seen dataset.
y_unseen = to_categorical(Y_unseen, Y_size + 1)                                 # One-Hot Encoding on Y unseen dataset.


# 2.5) Defining seen (including training and validation dataset) and testing datasets;
x_seen, y_seen = x_seen, y_seen                                                 # Training and validating dataset (seen).
x_test, y_test = x_unseen, y_unseen                                             # Testing dataset (unseen). The column limit [:,9:] is to exclude parameters.                                                     
                                                        
# Section 3: Neural Network 
# 3.1) The parameters under investigation and their values;

'''CHANGE number of nodes HERE'''
'''To change the number of nodes and other hyperparameters'''
num_nodes = 100

initilizer = 'normal'                                                           # Initializer that generates tensors with a normal distribution.
activation = 'relu'                                                             # Rectified linear unit (relu).
dropout_size = 0.1                                                              # Technnique for randomly selected nodes are ignored during training.
constraint_num = 1
optimizer = Adam                                                                # Adaptive moment estimation (Adam).
learning_rate = 0.01
decay_rate = 0
batch_size = 40                                                                 # The number of training instances shown to the model before a weight update is performed.
epochs = 100                                                                     # The number of times that the model is exposed to the training dataset.

# 3.2) The structure of the neural network;
input_size = X_size                                                             # The input size of the neural network
output_size = (Y_size + 1)                                                      #The output size of the neural network

model = Sequential()                                                            # Neural Network by using Sequential structure.
model.add(Dense(num_nodes,                                                      # Neural network model with 1 hidden layer.
                    kernel_initializer=initilizer,
                    activation=activation, 
                    input_shape=(input_size,)))                                 # Input shape represents the number of nodes in input layer.
model.add(Dropout(dropout_size))                                                # The dropout after the hidden layer and before output layer.
model.add(Dense(output_size, activation='softmax',                              # The output layer, Y_size the number of nodes in the output.
                    kernel_constraint= maxnorm(constraint_num)))
model.compile(loss='categorical_crossentropy',                                  # Compile the model, categorical_crossentropy is the objective function for classification.
                    optimizer=optimizer(lr=learning_rate, decay = decay_rate ),
                    metrics=['accuracy'])                                       # Accuracy is the only available metric in keras at the moment.

# 3.3) Training the neural network using training data;
Training = model.fit(x_seen, y_seen,                                            # Train the neural network with the training dataset. This step is also know as weights optimisation.
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.25)                                      # Split 25% data to validation dataset (represent 60%  for training, 20% for validation and 20% for testing from original data)

# 3.4) Checking the size of training, validation and test datasets;
validation_size = Training.validation_data[0].shape[0]                          # To check the size (number) of data in validation dataset (decided size is 20% from original data).
training_size = x_seen.shape[0] - validation_size                               # To check the size (number) of data in training dataset (decided size is 60% from original data).
testing_size = x_test.shape[0]                                                  # To check the size (number) of data in testing dataset (decided size is 20% from original data).

# 3.5) Tracing training and validation datasets (this can be done after (3.4) is defined);
X_training, Y_training = X_seen[0:training_size], Y_seen[0:training_size]       # Training data (raw - not transformed, scaled)
X_validation, Y_validation = X_seen[training_size:], Y_seen[training_size:]     # Validation data (raw - not transformed, scaled)

# Section 4: Evaluating the neural network model.

# 4.1) Evaluating the performance of the neural network;
test_score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)   # Evaluate the metric (performance) of the neural network in terms of accuracy and loss

# 4.2 Predicting using testing dataset;
prediction = model.predict(x_test, verbose = 0, batch_size= batch_size)         # Predict the class (1-8) for the given inputs (x).

# 4.3 The result for prediction;
#convert binary class matrices to class vectors
y_actu = [np.argmax(y_test, axis=1)]                                            # Convert binary class matrices to class vectors  (reverse One-Hot Encoding) on y actual.
y_pred = [np.argmax(prediction, axis=1)]                                        # Convert binary class matrices to class vectors  (reverse One-Hot Encoding) on y prediction.

#SECTION 5: Result evaluation (Confusion matrix and history plot)
# 5.1) Confusion Matrix;
confusion_mtrx = pd.crosstab(y_actu, y_pred, rownames=['Actual'],               # Create the Confusion Matrix from y_actu and y_pred.
                                           colnames=['Predicted'], margins=True)

# 5.2) Confusion matrix (normalised);
confusion_norm_mtrx = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize = 'index')         # Normalise by dividing with total actual. 

#SECTION 6: Result evaluation. 
# 5.1) Confusion Matrix;
plt.figure(1)
cm = confusion_mtrx
sn.heatmap(cm, annot=True, cmap=plt.cm.Blues, linewidths=.5, fmt="d", square = True)
plt.show()

# 5.2) Confusion matrix (normalised);
plt.figure(2)
cm_norm = confusion_norm_mtrx
sn.heatmap(cm_norm, annot=True, vmin=0, vmax=1, cmap=plt.cm.gray_r, linewidths=.5, square = True)
plt.show()

# 5.3) Plot of accuracy history;
plt.figure(3)                                                                   # Accuracy plot
plt.plot(Training.history['acc']) 
plt.plot(Training.history['val_acc'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('Accuracy (-)')
plt.xlabel('Epoch (-)')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# 5.4) Plot for loss history;
plt.figure(4)                                                                   # Loss plot
plt.plot(Training.history['loss'])
plt.plot(Training.history['val_loss'])
plt.title('Loss vs Epoch')
plt.ylabel('Loss (-)')
plt.xlabel('Epoch (-)')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Section 7: Importing the result to an Excel file.
# 7.1) The directory to save the Excel file;

'''CHANGE File directory HERE to SAVE the output files'''
os.chdir('C:/Users/enry_/OneDrive/UNI/Python/Results')        

# 7.2) A function to import the data;
def excelsave(array,filename): 
    
    wb=Workbook()

    ws=wb.active
    
    for i in range(len(array)):
        for j in range(len(array[0])):
            ws.cell(row=i+1, column=j+1, value=array[i,j])
        
    wb.save(filename)
    
    return


pact = np.array([[]])

ppred= np.array([[]])

for i in range(200):
    pact  = np.append(pact,np.array([[prediction[i][y_actu[0][i]]]]),axis=1)    # output for the real model
    
    ppred = np.append(ppred,np.array([[prediction[i][y_pred[0][i]]]]),axis=1)   # output for predicted model
    
    out = np.append(pact,ppred,axis=0)


    
error_classification =np.divide(y_pred,y_actu, dtype =float)                    # To detect wrong classification, 1 is correct, else is wrong.

append_output = np.append(np.transpose(np.append(X_unseen,prediction,axis=1)), 
                          np.append(y_actu, y_pred, axis=0), axis=0)            # Append all the columns.

array_to_export = np.transpose(np.append(np.append(append_output,                         # Testing dataset and results to export.
                                                 error_classification, axis=0),
                                                    out, axis=0))

array_to_export1 = np.transpose(np.append(np.transpose(X_training),             # Training dataset to export.
                                      np.reshape(Y_training, (1,600)), axis=0))   
array_to_export2 = np.transpose(np.append(np.transpose(X_validation),           # Validation dataset to export.
                                      np.reshape(Y_validation,(1,200)), axis=0))



'''CHANGE NAME HERE to SAVE different output files'''
'''This part is important to change the name of excel files'''

excelsave(array_to_export, 'StandardTest.xlsx')                                # File name for testing and results.
excelsave(array_to_export1, 'StandardTrain.xlsx')                              # File name for training dataset.
excelsave(array_to_export2, 'StandardValidation.xlsx')                         # File name for validation dataset.

# Section 8: The python output.
print
print('1. The Neural Network Structure summary: ')
print(model.summary())                                                          # Show the summary of the neural network model.
print
print('2. The Neural Network is trained and validated on %r training and %r validation samples' 
           % (training_size, validation_size))
print('   The Neural Network is tested on %r testing samples' % (testing_size))
print
print('3. The accuracy and loss: ')                                             # Show the accuracy and loss metrics (the performance).
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
print
print('4. The Confusion Matrix: ')
print(confusion_mtrx)                                                           # Show the Confusion Matrix.
print
print('5. The Normalised Confusion Matrix: ') 
print(confusion_norm_mtrx)                                                      # Show the Normalised Confusion Matrix.
