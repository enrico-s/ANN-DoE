# Import python packages.

import numpy as np
from scipy import integrate
from scipy.optimize import differential_evolution
import gc
import pandas as pd

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras import backend
from sklearn.model_selection import train_test_split

import time



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





def neural_network(X, Y):

    '''
    Trains and tests an artificial neural network.

    Input:
        X: dataset fed to the ANN
        Y: labels for classification

    Output:
        ANN test accuracy

    '''

    # 1) The size of the dataset, also represents the visible structure of the neural network;

    '''CHANGE Input Size HERE'''

    '''This part is need to adjust when the data is changing'''

    X_size = np.shape(X)[1]  # The input size of the neural network.

    Y_size = 8  # The output size of the neural network.



    # Section 2: Preprocessing the data. 

    # 2.1) Fix random seed for reproducibility;

    seed_num = 8

    # 2.2) Spliting data into 'seen' (training & validation) and 'unseen' (testing);

    X_seen, X_unseen, Y_seen, Y_unseen = train_test_split(X, Y, test_size=0.2,

                                                          random_state=seed_num)  # Spliting X & Y data into seen and unseen datasets.


    # 2.3) Scaling for the X dataset;
    
    scaler = StandardScaler()  # Set the scaling feature - removing the mean and scaling to unit variance.
    scaler.fit(X_seen, X_unseen)  # Set the scaling tool only on X data only.
    x_seen = scaler.transform(X_seen)  # Transform the seen (training & validation) data using the scaler.
    x_unseen = scaler.transform(X_unseen)  # Transform the unseen (testing) data using the scaler.



    # 2.4) Convert class vectors to binary class matrices (One-Hot Encoding) for Y dataset;

    y_seen = to_categorical(Y_seen, Y_size + 1)  # One-Hot Encoding on Y seen dataset.
    y_unseen = to_categorical(Y_unseen, Y_size + 1)  # One-Hot Encoding on Y unseen dataset.



    # 2.5) Defining seen (including training and validation dataset) and testing datasets;

    x_seen, y_seen = x_seen, y_seen  # Training and validating dataset (seen).
    x_test, y_test = x_unseen, y_unseen  # Testing dataset (unseen). The column limit [:,9:] is to exclude parameters.

  
    # Section 3: Neural Network 
    # 3.1) The parameters under investigation and their values;
    '''CHANGE number of nodes HERE'''
    '''To change the number of nodes and other hyperparameters'''

    num_nodes = 100

    initilizer = 'normal'  # Initializer that generates tensors with a normal distribution.
    activation = 'relu'  # Rectified linear unit (relu).
    dropout_size = 0.1  # Technnique for randomly selected nodes are ignored during training.
    constraint_num = 1
    optimizer = Adam  # Adaptive moment estimation (Adam).
    learning_rate = 0.01
    decay_rate = 0
    batch_size = 40  # The number of training instances shown to the model before a weight update is performed.
    epochs = 100  # The number of times that the model is exposed to the training dataset.

    # 3.2) The structure of the neural network;

    input_size = X_size  # The input size of the neural network
    output_size = (Y_size + 1)  # The output size of the neural network


    model = Sequential()  # Neural Network by using Sequential structure.
   
    model.add(Dense(num_nodes,  # Neural network model with 1 hidden layer.
                    kernel_initializer=initilizer,
                    activation=activation,
                    input_shape=(input_size,)))  # Input shape represents the number of nodes in input layer.

    model.add(Dropout(dropout_size))  # The dropout after the hidden layer and before output layer.
    
    model.add(Dense(output_size, activation='softmax',  # The output layer, Y_size the number of nodes in the output.
                    kernel_constraint=maxnorm(constraint_num)))

    model.compile(loss='categorical_crossentropy',
                  # Compile the model, categorical_crossentropy is the objective function for classification.
                  optimizer=optimizer(lr=learning_rate, decay=decay_rate),
                  metrics=['accuracy'])  # Accuracy is the only available metric in keras at the moment.

    # 3.3) Training the neural network using training data;
    model.fit(x_seen, y_seen,
              # Train the neural network with the training dataset. This step is also know as weights optimisation.
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_split=0.25)  # Split 25% data to validation dataset (represent 60%  for training, 20% for validation and 20% for testing from original data)


    # Section 4: Evaluating the neural network model.
    # 4.1) Evaluating the performance of the neural network;
    test_score = model.evaluate(x_test, y_test, verbose=0,
                                batch_size=batch_size)  # Evaluate the metric (performance) of the neural network in terms of accuracy and loss

    # Clears the ANN models from the memory for enhancing the DoE optimiser speed
    del model
    gc.collect()
    backend.clear_session()
    
    return test_score[1]



def new_dataset(exp_cond, models, parameters, labels, sigma):
    '''
    It generates the dataset used to feed the ANN.

    Input:
        exp_cond: 2n array ([x0_1,...x0_n, u1,..un]) , with initial states (x0) and time-independent inputs (u)  

    Output:
        X: dataset for training the ANNs.
    '''
    
    n_exp = int(len(exp_cond) / 4)  # number of experiments
    n_samples = 2 #int(len(t))         # numer of samples per experiment
    
    des_X = np.array([np.concatenate([ np.concatenate([
    
                measurement(np.array([exp_cond[i], 0 , 0 ]), exp_cond[i + n_exp],
                                    exp_cond[i+2*n_exp] , sigma, parameter, models[label - 1]) , 
            
                measurement(np.array([exp_cond[i], 0 , 0 ]), exp_cond[i + n_exp],
                                    exp_cond[i+2*n_exp]+60+exp_cond[i+3*n_exp] , sigma, parameter, models[label - 1]) ,
            
                       ]) #     for j in range(n_samples) ])                      
                for i in range(n_exp) ])
              
        for label, parameter in zip(labels, parameters) ])
    
    #prel_X = np.load('temporary.npy')  
    #X = np.concatenate([prel_X, des_X], axis=1)
    
    return des_X




def objective_function(exp_cond, equation_codes, parameters, labels, x0, sigma):
    '''
    Objective function to be minimized.

    Input:

        exp_cond: 2n array ([u1,..un,t1,..tn]) with (temperatures and sampling times) to be optimized

    Output:

        OF: inverse of the ANN test accuracy.

    '''

    models = [compile(model, "<string>", "eval") for model in equation_codes]
    
    start_time = time.time()

    X = new_dataset(exp_cond, models, parameters, labels, sigma)

    intermediate_time = time.time()

    OF = - neural_network(X, labels)

    final_time = time.time()
    
    gc.collect()

    print(

        'Generation of dataset took {0:.2f}s; Traininig of Network took {1:.2f}s'.format(intermediate_time - start_time,

                                                                                    final_time - intermediate_time))



    return OF





if __name__ == "__main__":
    
    __spec__ = None
    
    '''
    DEFINE the candidate kinetic models as a list of string-type elements.
    np.inner(,) makes the scalar product between the stoichiometric array and the kinetic factors.
    '''
    
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
    

    # Fixed experimental conditions
    '''CHOOSE the fixed sampling times and the experimental error.
    
        If sampling time is wanted to be free or other exp variables to be fixed, the code must be modified.
        In particular, the bounds for DoE optimisation ;
                       the input provided to 'differential_evolution' , 'objective_function' , and 'new_dataset' functions ;
                       and MOST IMPORTANT, in 'new_dataset' how the experimental conditions are passed to the function 'measurement'.
    '''
        
    x0 = np.array([ [100, 0, 0]   ])  # initial concentration of A B and C

    #u = np.array([  680, 599, 705, 640, 710, ])  # temperatures at which the experiments are conducted

    t = np.array([  100, 200, 350 ])# sampling times for each experiment

    sigma = np.array([0.00, 1.00])  # noise on the concentration measurements [relative, constant]

    
    #Import the kinetic paramter values and associated label from .csv file
    '''NOTE. Set the right path for the file with model parameters'''
    
    parameters_df = pd.read_csv('C:/Users/enry_/OneDrive/UNI/Python/Data/ModelParameters.csv')
    
    parameters_df.label = parameters_df.label.astype(int)
    labels_set = parameters_df['label'].values
    parameters_set = [parameters_df.loc[n].values[2:] for n in range(len(parameters_df.index))]


    #Conduct the preliminary experiments (already designed) and store the dataset then recalled by the function 'new_dataset'
    #n_exp = int(len(u))
    #n_samples = int(len(t))
    #
    #prel_X = np.array([np.concatenate([ np.concatenate([
    #
    #            measurement(x0[i], u[i], t[j], sigma, parameter, models[label - 1])
    #        
    #                        for j in range(n_samples) ])                      
    #            for i in range(n_exp) ])
    #          
    #    for label, parameter in zip(labels_set, parameters_set) ])
    #    
    #np.save('temporary.npy', prel_X)
    #
    
    '''
    SET the bounds for the DoE variables: [(cA0),... ,
                                            (temperature)....]
                                            
    The the number of experiments is related to the number of elements in the list.
    '''
    
    bounds = [  (0 , 250),  (0 , 250),  
    
                (520, 720),  (520, 720), 
                
                (50, 290), (50, 290),
                
                (0,240), (0,240), ] 
   
   

  
    # Find the optimal DoE with the differential evolution (DE) algorithm
    ''' The DE parameters could be adjusted.
        These values shown a good behaviour in terms of results quality and computational time.
        workers=-1 sets multiprocessing computation.
    '''
    start = time.time()
  
    np.random.seed(4)
          
    optimal_DoE = differential_evolution(objective_function, bounds=bounds, args=(equation_codes, parameters_set, labels_set, x0, sigma),

                                      strategy='best1bin', maxiter=30, popsize=15, tol=0.1, mutation=(0.5, 1),

                                      recombination=0.7 , updating='deferred' , workers=-1 )
                             
    finish = time.time()                                                                                               
 
 
    n_experiments = int(len(bounds)/4)     
                                         
    print('The optimal experimental conditions are:')

    for i in range(n_experiments):
        print('    Experiment number: %d ' %(i+1))
                
        print('    Concentration of A: %f mol/m3' % optimal_DoE.x[i])
    
        print('    Temperature: %f K' % optimal_DoE.x[i+n_experiments])
        
        print('    First sample at: %f sec' % optimal_DoE.x[i+2*n_experiments])
        
        print('    Second sample at: %f sec' % optimal_DoE.x[i+3*n_experiments])
    
        print('')
        

    print('The value of ANN test accuracy achieved is: %f' % (-optimal_DoE.fun))
    
    print('The time required for optimisation process is %f seconds' %(finish-start))