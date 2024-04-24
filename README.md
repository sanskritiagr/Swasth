# SwasthAI

## Description
SwasthAI is an application designed to assist users in predicting diseases and recommending specialists based on their symptoms. Leveraging machine learning models, the application analyzes user input to provide accurate predictions and match users with relevant healthcare professionals. Additionally, the application utilizes doctors' data and patient ratings to ensure personalized recommendations tailored to each user's needs.

## Table of Contents

1. [Description](#Description)
2. [How to run the file](#Howtorunthefile)
3. [Installation](#installation)
4. [Data](#data)
5. [Usage](#usage)
6. [Pipeline](#pipeline)
7. [Models Used](#modelsused)
8. [Accuracy](#accuracy)



## How to run the file
The project is already deployed on https://swasth.streamlit.app/ 

## Installation

To install and run SwasthAI locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/sanskritiagr/SwasthAI.git
   ```

2. Navigate to the project directory:

   ```bash
   cd SwasthAI
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

7. Access the application in your web browser at `http://localhost:8501`.

## Usage

1. Choose your symptoms.
2. Click the "Submit" button to generate predictions for the disease and recommended specialist.
3. Then login(if you are an existing user) using a patient-id(values from 1-1000) and password:123. If you are a new user simply choose New User.
4. Now, you can see all the doctors available.

## Data
The data used are:
1. [Symptoms Mapping: ](https://github.com/sanskritiagr/SwasthAI/blob/main/data.csv) It contains diseases mapped with the symptoms and the specialist to consult. 
2. [Doctor Table: ](https://github.com/sanskritiagr/SwasthAI/blob/main/doc_table.csv) It contains all the information about the doctor including their name, specialty, phone, hospital in which they work, their opening and closing time along with the latitudes to their location. The table is generated, for which the code can be viewed in this [file](#https://github.com/sanskritiagr/SwasthAI/blob/main/Generate_Doc_Table.ipynb).
3. [Rating Table: ](https://github.com/sanskritiagr/SwasthAI/blob/main/rating.csv) It is the table of various patients registered with our application and the ratings they gave for doctors they have consulted in the past. The data is generated, for which the code can be viewed in this [file](#https://github.com/sanskritiagr/SwasthAI/blob/main/Rating_Table_Gen.ipynb)


## Pipeline

1. Takes symptoms as input from the user. Stores it in the form of list.
2. Passes this list to a function(predict_disease from [disease_predict.py](https://github.com/sanskritiagr/SwasthAI/blob/main/disease_predict.py) and predict from [specialist_predict.py](https://github.com/sanskritiagr/SwasthAI/blob/main/specialist_predict.py)) which outputs the disease and the specialist.
3. Then we login using our patient-id(values from 1-1000) and password(123). We can also choose 'New user'.
4. If new user, then we tell them ratings of doctors based on previous users'/patients' ratings.
5. If an existing user, we tell him ratings personalized for him. The functions for which are present in [doctor_rating.py](https://github.com/sanskritiagr/SwasthAI/blob/main/doctor_rating.py)


![image](https://github.com/sanskritiagr/SwasthAI/assets/96240350/bd1a8e75-1705-4ff3-bb1a-86847adfcc5f)

## Models used
1. kNN, Random Forest Classifier, Multinomial Naive Bayes' Classifier, and Support Vector Machine are used to predict the disease and specialist.
2. We have also ensembled these models using hard voting.
3. Also a model is created for collaborative filtering of doctors for various patients.(see doctor_rating.py file to learn more about it)

## Accuracy 
We got 93.9% accuracy for specialist recommendation and 88.9% for disease prediction.




