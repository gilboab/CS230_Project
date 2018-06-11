This folder contains the project code
Content:
1. bittrex.py - API from bittrex
2. collect_data.py - our code that runs and collect data every min. 
                     There are some primitive tricks to increase stability of this file.
                     because it is dealing with a web server over the internet, sometimes it does not get response on time and crash. To 
                     overcome this problem we run few instances of the same code in parallel one instance is collecting the data and the 
                     other instances are ready to take its place once it crashes. For that wee maintain a watch dog file between the
                     instances and list file
3. predictor.py - Main file presented the code up to the milestone
4. feature_collector.py - the file that takes the raw data files and create and saves features and labels.
5. predictor_dynamic_rnn.py - TensorFlow model for our dynamic RNN model
6. predictor_keras_th_single.py - the main categorical multi label model that we have built and presented in the report
7. predictor_keras_th_single_bidirectional.py - same model as above but configured to work in bidirectional RNN
8. predictor_rnn.py - Our Basic RNN model in Tensorflow that we described in the report
9. predictor_single_input_as_time_steps.py - the TensorFlow model we have built for modeling the feeding the features one by one to the RNN network
10. readme.txt - this file
11. simulate_results.py - the file used to simulate the trading model that use the predictions
12. tf_utils.py - python utility functions taken from one of the class HW
