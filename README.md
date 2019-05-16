# Disaster Response Pipeline Project

## Installation
In order to execute the application you should have **python 3.6.6** installed. 
The application needs plotly, flask, sqlalchemy, pickle, scikit-learn, nltk, matplotlib, numpy and pandas libraries. These are available as part of Anaconda installation **Anaconda 4.5.11**.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Overview
This web app can be used by an emergency response team during a disaster event (e.g. cyclone or earthquake), to classify a disaster messages into several categories. Based on the category classification, the message can be directed to the appropriate department/aid agency.

The application uses a machine learning model to categorize the messages. The code includes an ETL pipeline to parse, clean and load the messages into an SQLite database. This data is then processes by an ML pipeline to train the model. The trained model is serialized for usage in the web application.

## Project Structure

```
/
├── app/
│   ├── run.py                  #Application main file
│   └── templates/
│       ├── go.html             #page for displaying predicted classification
│       └── master.html         #application landing page
├── data/
│   ├── disaster_categories.csv #category data
│   ├── disaster_messages.csv   #messages data
│   ├── DisasterResponse.db     #Generated SQLite DB through ETL pipeline
│   ├── process_data.py         #ETL pipeline
├── models/
│   ├── classifier.pkl          #pickled trained model
│   ├── train_classifier.py     #ML Pipeline
│   └── workspace_utils.py      #utility file to keep workspace alive
├── README.md                   #README (this file)
├── ETL-pipeline.ipynb          #Jupyter notebook containing ETL pipeline discussion
└── ML-pipeline.ipynb           #Jupyter notebook containing ML pipeline discussion
```
* **data/process_data.py**: ETL Pipeline: Takes message data (input text) and message categories (labels) as input in CSV format. The data is merged, cleaned and formatted before loading it into an SQLite database.
* **model/train_classifier.py**: ML Pipeline: Takes the SQLite database produced in previous step as an input to train a ML model for message classification. The fitted model is stored in a pickle file.

## Further improvements
After the trained model performs the prediction and results are displayed, the application can allow user to fix the labels and submit.
This event can add the message and category data into the CSV input files and trigger the ETL pipeline and ML training process asynchronously.

## Acknowledgements
This app was built as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 
Code templates, utility files and data were provided by Udacity.