# Complete CNN + BiLSTM model integration with dual-stream architecture

import random
random.seed(2021)
from numpy.random import seed
seed(2021)
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(2021)

import random as python_random
python_random.seed(2021)

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import IPython
from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperModel
from numpy import array
from keras.models import Sequential, Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, Input, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import time
from contextlib import redirect_stdout

os.environ['TF_DETERMINISTIC_OPS'] = '0'

# Parameters
SCALE_ON = 1
OutputsFolderName = "Re_Verification_Austrilia_dataset"
window_size = [5, 8, 11]
forecast_horizon =10
NUM_FEATURES =1 #1
MaxTrials = 30
EpochsTuning =5
EpochsTraining = 200
ValidationSplit = 0.2
PATIENCE1 = 10
PATIENCE2 = 50



REMOVE_THIS_MANY_ROWS_FROM_END = 1

# Create necessary directories for outputs
parentDirectory = os.path.dirname(os.path.dirname(__file__))
outFolder = os.path.join(parentDirectory, "out", OutputsFolderName)
historydir = os.path.join(outFolder, 'history')
modelsdir = os.path.join(outFolder, 'models')
summarydir = os.path.join(outFolder, 'summary')
predictionsdir = os.path.join(outFolder, 'predictions')

if not os.path.exists(outFolder):
    os.makedirs(outFolder)
    os.makedirs(historydir)
    os.makedirs(modelsdir)
    os.makedirs(summarydir)
    os.makedirs(predictionsdir)

# Load data
DataFile = "AUS_ERPS_1991to2016_withSummedRemainder.csv"
data_file_path = os.path.join(parentDirectory, "data", DataFile)
dataset_train_full = pd.read_csv(data_file_path)

columns = dataset_train_full.columns
areaNames = dataset_train_full.iloc[:, 2]

# Define actual forecast period for calculating errors
ActualDataStart = columns.get_loc("2007_SA2")
ActualDataEnd = columns.get_loc("2016_SA2")
ActualTruth = dataset_train_full.iloc[:, ActualDataStart:(ActualDataEnd + 1)]

# Data for training (until the jump-off year)
from_Column_ERPs = columns.get_loc("1991_SA2")
to_Column_ERPs = columns.get_loc("2006_SA2")
dataset_for_forecasts = dataset_train_full.iloc[:, from_Column_ERPs:(to_Column_ERPs + 1)]

# Transpose the array so each column is a small area
y_full = dataset_for_forecasts.T
y_multiSeries = pd.DataFrame(y_full[:(len(y_full))])

# Remove remainder row from training data
y_multiSeries = y_multiSeries.iloc[:, :(y_multiSeries.shape[1] - REMOVE_THIS_MANY_ROWS_FROM_END)]
y_multiSeries = np.array(y_multiSeries)

# Keep the original dataset for scaling purposes
OriginalDataForScaling = y_multiSeries.copy()

# Debugging: check the shape of the data
print("Shape of y_multiSeries:", y_multiSeries.shape)

# Functions for data transformation
def lstm_data_transform(x_data, y_data, num_steps, forecast_horizon):
    X, y = [], []
    for i in range(x_data.shape[0]):
        end_ix = i + num_steps
        if end_ix - 1 >= x_data.shape[0]:
            break
        seq_X = x_data[i:end_ix]
        X.append(seq_X)
    x_array = np.array(X)
    return x_array

def lstm_full_data_transform(x_data, y_data, num_steps, forecast_horizon, scale_on, OriginalDataForScaling):
    X, y = [], []
    print("x_data shape before transformation:", x_data.shape)

    if scale_on == 1:
        x_data = (x_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())
        y_data = (y_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())

    for j in range(x_data.shape[1]):
        for i in range(x_data.shape[0]):
            end_ix = i + num_steps
            if end_ix >= x_data.shape[0]:
                break
            seq_X = x_data[i:end_ix, j]
            seq_y = y_data[end_ix, j]
            X.append(seq_X)
            y.append(seq_y)
    x_array = np.array(X)
    y_array = np.array(y)
    x_array = x_array.reshape(x_array.shape[0], x_array.shape[1], 1)
    return x_array, y_array

# Data transformation for validation
def lstm_full_data_transform2(x_data, y_data, num_steps, forecast_horizon, scale_on, OriginalDataForScaling):
    X, y, Xval, Yval, Xtrain, Ytrain = [], [], [], [], [], []
    nVal = 3

    if scale_on == 1:
        x_data = (x_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())
        y_data = (y_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())

    for j in range(x_data.shape[1]):
        for i in range(x_data.shape[0]):
            end_ix = i + num_steps
            if end_ix >= x_data.shape[0]:
                break
            seq_X = x_data[i:end_ix, j]
            seq_y = y_data[end_ix, j]
            X.append(seq_X)
            y.append(seq_y)

        # Keep the last nVal sequences for validation
        Xval.append(X[-nVal:])
        Yval.append(y[-nVal:])
        Xtrain.append(X[:-nVal])
        Ytrain.append(y[:-nVal])
        X, y = [], []

    x_array = np.array(Xtrain).reshape(-1, num_steps, 1)
    y_array = np.array(Ytrain).reshape(-1)
    Xval = np.array(Xval).reshape(-1, num_steps, 1)
    Yval = np.array(Yval).reshape(-1)
    return x_array, y_array, Xval, Yval

# Callback for clearing outputs after training
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

# Function to calculate Median Absolute Percentage Errors
def calcMAPEs(TheForecast, ActualTruth):
    MAPES = np.abs(np.array(TheForecast) - np.array(ActualTruth)) / np.array(ActualTruth) * 100
    medianMapes = np.median(MAPES, axis=0)
    meanMAPES = np.mean(MAPES, axis=0)
    MAPES_greater10 = np.count_nonzero(MAPES > 10, axis=0) / (MAPES.shape[0]) * 100
    MAPES_greater20 = np.count_nonzero(MAPES > 20, axis=0) / (MAPES.shape[0]) * 100
    ErrorSummary = pd.DataFrame([medianMapes, meanMAPES, MAPES_greater10, MAPES_greater20])
    ErrorSummary = ErrorSummary.rename(index={0: "medianMapes", 1: "meanMAPES", 2: " >10%", 3: " >20%"})
    OneLine = pd.DataFrame(np.hstack((medianMapes, meanMAPES, MAPES_greater10, MAPES_greater20)).ravel())
    MAPES = pd.DataFrame(MAPES)
    return MAPES, ErrorSummary, OneLine

#put error values into the forecasts so when create summary array the error values are already there
def ArraysWithErrors(OneLine,ModelTypeName,num_steps_w,forecast_horizon,TheForecast,MAPES_pd,ErrorSummary,areaNames):
    
    ConcatArr=pd.concat([TheForecast.reset_index(drop=True),MAPES_pd.reset_index(drop=True),ErrorSummary.reset_index(drop=True)],axis=1)
    
    #let's add a line with details into our summary forecasts
    df=pd.DataFrame(columns=ConcatArr.columns)
    df.append(pd.Series(name='info'))
    df.loc[0,:]=ModelTypeName+" "+str(num_steps_w)+" steps"
    df2=pd.concat([df,ConcatArr])

    newIndex=pd.concat([pd.Series(["info"]),areaNames])
    ind=pd.DataFrame(newIndex)
    ind=ind.rename(columns={ind.columns[0]:"SA2"})
    df2['SA2_names']=ind['SA2']
    df2.set_index('SA2_names',inplace=True)

    #Let's add details to the error summary so that when we aggregate them we know
    #which one is which
    df_errorArray=pd.DataFrame(columns=ErrorSummary.columns)
    df_errorArray.append(pd.Series(name='info'))
    df_errorArray.loc[0,:]=ModelTypeName+" "+str(num_steps_w)+" steps"
    ErrorSummary=pd.concat([df_errorArray,ErrorSummary])
    ErrorSummary=ErrorSummary.rename(index={0:"info"})
    
    return ErrorSummary,df2

#Function to perform forecasts
def runForecasts(dataset_train_full,num_features,num_steps,forecast_horizon,filename1,model1,SCALE_ON,OriginalDataForScaling,outFolder,areaNames):
    for c_f in range(len(dataset_train_full)):
        if (c_f % 100 == 0):
            print(c_f)
        if c_f==0:
            #We will measure execution time for each loop
            ETime=pd.DataFrame(index=range(len(dataset_train_full)), columns=['eTime'])
        
        start1=time.time()


        columns=dataset_train_full.columns
        from_Column_ERPs=columns.get_loc("1991_SA2")
        to_Column_ERPs=columns.get_loc("2006_SA2")
        
        y_full=dataset_train_full.iloc[c_f,from_Column_ERPs:(to_Column_ERPs+1)]
        y=y_full.reset_index(drop=True)
        Area_name=areaNames.iloc[c_f]
        ActualData=pd.DataFrame(y_full)
        ActualData=ActualData.rename(columns={"0":Area_name})
        ActualData_df=ActualData.T        

        if (SCALE_ON==1):
            y=(y-OriginalDataForScaling.min())/(OriginalDataForScaling.max()-OriginalDataForScaling.min())

        y=np.array(y)

        x_new = lstm_data_transform(y, y, num_steps=num_steps, forecast_horizon= forecast_horizon)
        
        x_train = x_new

        test_input=x_new[-1:]
        test_input_prescaled=test_input
        temp1=test_input

        test_input=test_input.reshape(1,num_steps,1)

        PredictionsList=list()
        LSTMPredictionsList=list()

        #Do the rolling predictions 
        for i in range(forecast_horizon):             
            test_input=test_input.reshape(1,num_steps,1)
            test_input=np.asarray(test_input).astype('float32')
            test_output = model1.predict(test_input, verbose=0)
            
            test_input[0,:(num_steps-1)]=test_input[0,1:]
            LSTMPredictionsList.append(test_output.reshape(1))
    
            test_input[0,(num_steps-1)]=test_output
            PredictionsList.append(test_output.reshape(1))
                
        PredictionsList=np.array(PredictionsList)
        predictions=PredictionsList.reshape(forecast_horizon,num_features)
        
        if (SCALE_ON==1):
            df_predict=predictions*(OriginalDataForScaling.max()-OriginalDataForScaling.min())+OriginalDataForScaling.min()

        else:
            df_predict=predictions
        
        #Let's label the predictions for each of the areas in a larger dataframe
        df_predict=pd.DataFrame(df_predict)
        
        for count_through_columns in range(num_features):
            df_predict=df_predict.rename(columns={df_predict.columns[count_through_columns]:Area_name})

        temp_predict_df=df_predict.T        
        
        if c_f==0:
            full_array_SA2s=temp_predict_df.iloc[[0]]

        else:
            frames_SA2=[full_array_SA2s, temp_predict_df.iloc[[0]]]
            full_array_SA2s=pd.concat(frames_SA2)

        endLoop=time.time()
        ETime.iat[c_f,0]=endLoop-start1

    timestr = time.strftime("%Y%m%d")
    FullArrayLocation=filename1+timestr+".csv"

    parentDirectory=os.path.dirname(os.path.dirname(__file__))
    file_path_df=os.path.join(outFolder,'predictions',FullArrayLocation)

    full_array_SA2s.to_csv(file_path_df)

    #Return the predictions
    return full_array_SA2s

from tensorflow.keras.layers import Layer
import tensorflow as tf

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: hidden state (BiLSTM output)
        # values: BiLSTM outputs (time steps)
        
        # Add the query dimension to match the shape of values
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Score calculation
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        
        # Softmax over the time dimension (axis=1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Multiply attention weights by values (BiLSTM outputs)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class CNN_BiLSTM_Attention_HyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # CNN stream
        # x_cnn = tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        #                                kernel_size=3, activation='relu')(inputs)
        # x_cnn = tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        #                                kernel_size=3, activation='relu')(x_cnn)
        # x_cnn = tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        #                                kernel_size=3, activation='relu', padding='same')(x_cnn)
        # x_cnn = tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        #                                kernel_size=3, activation='relu', padding='same')(x_cnn)
        

        #x_cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(x_cnn)
        
        
        x_cnn = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x_cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x_cnn)
        x_cnn = tf.keras.layers.Conv1D(filters=96, kernel_size=3, activation='relu', padding='same')(x_cnn)
        x_cnn = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x_cnn)

        
        
        
        
        
        
        x_cnn = tf.keras.layers.Flatten()(x_cnn)
        
       # I have comment this code due to ablation studies
        
        #BiLSTM stream
        
        
        
        lstm_out, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=hp.Int('units', min_value=128, max_value=512, step=32),
                                  return_sequences=True, return_state=True, activation='relu'))(inputs)
        
        # Apply Bahdanau Attention on the output of BiLSTM
        query = tf.keras.layers.Concatenate()([forward_h, backward_h])
        context_vector, attention_weights = BahdanauAttention(hp.Int('attention_units', min_value=32, max_value=128))(query, lstm_out)
        
        # Combine both CNN and attention outputs
        x = tf.keras.layers.concatenate([x_cnn, context_vector])
        
        #Dense layers
        x = tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=32),
                                  activation='relu')(x_cnn)
        x = tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=32),
                                  activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse', metrics=['mse'])
        
        return model



# Now we'll replace the LSTM models with the CNN+BiLSTM model in the tuning and training loop
ExperimentalInfo = []
SummaryArrayFlag = 0
ModelTypeName = "CNN_+Korea_Ablation"

for current_window in range(len(window_size)):
    if current_window > 0:
        del x_new_w, y_new_w
    
    # Reset the random seeds for every model run
    seed(2021)
    tf.random.set_seed(2021)
    python_random.seed(2021)
    random.seed(2021)

    num_steps_w = window_size[current_window]
    INPUT_SHAPE = (num_steps_w, NUM_FEATURES)

    # Timing for model tuning, training, and forecasting
    a = time.time()

    # Transform the data for validation and tuning
    x_new_w, y_new_w, x_new_w_VAL, y_new_VAL = lstm_full_data_transform2(y_multiSeries, y_multiSeries, num_steps_w, forecast_horizon, SCALE_ON, OriginalDataForScaling)

    # Initialize CNN + BiLSTM model
    # Use the CNN + BiLSTM + Attention model
    CNN_BiLSTM_Attention_hypermodel = CNN_BiLSTM_Attention_HyperModel(input_shape=INPUT_SHAPE)

    
    projectName = "CNN_BiLSTM_" + str(num_steps_w)
    
    bayesian_opt_tuner = BayesianOptimization(
        CNN_BiLSTM_Attention_hypermodel,
        objective='mse',
        max_trials=MaxTrials,
        seed=2021,
        executions_per_trial=1,
        directory=os.path.normpath('C:/keras_tuning2'),
        project_name=projectName,
        overwrite=True
    )

    # Tune the model
    bayesian_opt_tuner.search(x_new_w, y_new_w, epochs=EpochsTuning, validation_data=(x_new_w_VAL, y_new_VAL),
                              verbose=2, callbacks=[ClearTrainingOutput(),
                              keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE1)])

    # Get the best model
    bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
    Model1 = bayes_opt_model_best_model[0]

    # Reduce learning rate if validation error plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE1, min_lr=0.000005, verbose=2)

    # Train the model with the best hyperparameters
    history = Model1.fit(x_new_w, y_new_w, epochs=EpochsTraining, validation_data=(x_new_w_VAL, y_new_VAL),
                         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE2, mode='auto', restore_best_weights=True), reduce_lr],
                         verbose=2)

    hist_getBest = Model1.history.history['val_loss']
    n_epochs_best = np.argmin(hist_getBest) + 1

    # Split validation data into training and validation sets
    x_new_w2, x_new_w_VAL2, y_new_w2, y_new_VAL2 = train_test_split(x_new_w_VAL, y_new_VAL, test_size=ValidationSplit, random_state=2021)

    # Further reduce learning rate during additional training
    reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE1, min_lr=0.000001, verbose=2)

    # Define the folders for storing models, history, and predictions
    outFolder = os.path.join(parentDirectory, "out", OutputsFolderName)
    historydir = os.path.join(outFolder, 'history')
    modelsdir = os.path.join(outFolder, 'models')
    predictionsdir = os.path.join(outFolder, 'predictions')

    # Save the model
    timestr1 = time.strftime("%Y%m%d")
    ModelSaveName2 = ModelTypeName + "_" + str(num_steps_w)
    save_model_here = os.path.join(modelsdir, (ModelSaveName2))
    Model1.save(save_model_here)
    SaveHistoryHere = os.path.join(historydir, (ModelSaveName2 + ".csv"))

    # Save the training history
    history_df = pd.DataFrame(history.history)
    with open(SaveHistoryHere, mode='w') as f:
        history_df.to_csv(f)

    # Save model summary
    SaveSummaryHere = os.path.join(outFolder, "summary", (ModelSaveName2 + ".txt"))
    with open(SaveSummaryHere, 'w') as f:
        with redirect_stdout(f):
            Model1.summary()

    # Now let's run the forecasts
    TheForecast = runForecasts(dataset_for_forecasts, NUM_FEATURES, num_steps_w, forecast_horizon, ModelSaveName2, Model1, SCALE_ON, OriginalDataForScaling, outFolder, areaNames)

    # Calculate errors and create error summary
    MAPES, ErrorSummary, OneLine = calcMAPEs(TheForecast, ActualTruth)
    OneLine = OneLine.rename(columns={OneLine.columns[0]: ModelTypeName + "_" + str(num_steps_w) + "_steps_" + str(forecast_horizon) + "_year"})
    ErrorSummary, df2 = ArraysWithErrors(OneLine, ModelTypeName, num_steps_w, forecast_horizon, TheForecast, MAPES, ErrorSummary, areaNames)

    # Save the error summary to the predictions folder
    ErrorSummary_df = pd.DataFrame(ErrorSummary.copy())
    ErrorSummaryFilePath = os.path.join(predictionsdir, (ModelSaveName2 + "ErrorSummary.csv"))
    with open(ErrorSummaryFilePath, mode='w') as f:
        ErrorSummary_df.to_csv(f)

    # Handle full forecast array and error summary
    if SummaryArrayFlag == 0:
        SummaryArrayFlag = 1
        FullForecastArray = df2
        FullErrorArray = OneLine
    else:
        FullForecastArray = pd.concat([FullForecastArray, df2], axis=1)
        FullErrorArray = pd.concat([FullErrorArray, OneLine], axis=1)

    # Record time taken for the run
    b = time.time()
    c = b - a
    The_Learning_rate = tf.keras.backend.eval(Model1.optimizer.lr)

    print('Window length: ' + str(num_steps_w) + ', Time taken: ' + str(c) + ' seconds')

    ModelConfig = Model1.optimizer.get_config()

    OurExperiments = [ModelTypeName, ModelSaveName2, forecast_horizon, num_steps_w, NUM_FEATURES, c, ModelSaveName2, outFolder,
                      MaxTrials, EpochsTuning, EpochsTraining, ValidationSplit, PATIENCE1, PATIENCE2, The_Learning_rate,
                      SCALE_ON, REMOVE_THIS_MANY_ROWS_FROM_END, n_epochs_best, np.array(ModelConfig)]
    ExperimentalInfo.append(OurExperiments)
    
    # Clear the session to avoid memory overflow
    del Model1, CNN_BiLSTM_Attention_hypermodel
    keras.backend.clear_session()

# Save the list of experiments
import csv

ExperimentalHistorySaveName = os.path.join(outFolder, 'Summary.csv')
with open(ExperimentalHistorySaveName, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(ExperimentalInfo)

FullPredictionsTogetherPath = os.path.join(outFolder, 'AllPredictions.csv')
FullForecastArray.to_csv(FullPredictionsTogetherPath)

FullErrorArrayTogetherPath = os.path.join(outFolder, 'ErrorSummary.csv')
FullErrorArray.to_csv(FullErrorArrayTogetherPath)

print("Experiment completed successfully.")

# %%
