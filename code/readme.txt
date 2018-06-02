This folder contains the project code
Content:
1. bittrex.py - API from bittrex
2. collect_data.py - our code that runs and collect data every min. 
                     There are some primitive tricks to increase stability of this file.
                     because it is dealing with a web server over the internet, sometimes it does not get response on time and crash. To 
                     overcome this problem we run few instances of the same code in parallel one instance is collecting the data and the 
                     other instances are ready to take its place once it crashes. For that wee maintain a watch dog file between the
                     instances and list file
3. predictor.py - the  main file with our model, the data set manipulation and preperation, some  printing methods and more
