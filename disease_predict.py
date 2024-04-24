import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

import numpy as np

import itertools

data = pd.read_csv('./data.csv')

disease_symp = data.drop('Specialist', axis=1)

disease_symp.head()

disease_symp['Symptoms'] = disease_symp['Symptoms'].str.lower()

symptoms_dummies = disease_symp['Symptoms'].str.get_dummies(sep=', ')
# symptoms_dummies = symptoms_dummies.rename(columns={col: col.strip() for col in symptoms_dummies.columns})

# Concatenate the dummy variables with the original DataFrame
df = pd.concat([disease_symp, symptoms_dummies], axis=1)


# Drop the original 'Symptoms' column
df.drop(columns=['Symptoms'], inplace=True)


def generate_new_examples(df):
    new_examples = []
    for index, row in df.iterrows():
        disease = row['Disease']
        symptoms = row.drop('Disease').index[row.drop('Disease') == 1].tolist()
        k = len(symptoms)
        for subset_length in range(k-2, k+1):
            subsets = itertools.combinations(symptoms, subset_length)
            for subset in subsets:
                new_example = {
                    'Disease': disease,
                    'Symptoms': ', '.join(subset)
                }
                new_examples.append(new_example)
    return pd.DataFrame(new_examples)

gen_data = generate_new_examples(df)


# df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Split the string representation of symptoms into dummy variables (1 if present, 0 otherwise)
symptoms_dummies = gen_data['Symptoms'].str.get_dummies(sep=', ')

# Concatenate the dummy variables with the original DataFrame
gen_data = pd.concat([gen_data, symptoms_dummies], axis=1)

# Drop the original 'Symptoms' column
gen_data.drop(columns=['Symptoms'], inplace=True)


gen_data = gen_data.sample(frac=1).reset_index(drop=True)


x = gen_data.drop('Disease', axis=1)
y = gen_data.Disease

x_train, x_test, y_train , y_test = train_test_split(x ,y, random_state = 50)

svc = SVC(kernel='linear', probability = True)

# Train the model
svc.fit(x_train, y_train)
 # Test the model
predictions = svc.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"SVM Accuracy: {accuracy}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)
print(f"SVM Confusion Matrix:")
print(np.array2string(cm, separator=', '))

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rfc.fit(x_train, y_train)
# Test the model
predictions = rfc.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"RandomForest Accuracy: {accuracy}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)
print(f"RandomForest Confusion Matrix:")
print(np.array2string(cm, separator=', '))

mnb =  MultinomialNB()
# Train the model
mnb.fit(x_train, y_train)
# Test the model
predictions = mnb.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"MultinomialNB Accuracy: {accuracy}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)
print(f"MultinomialNB Confusion Matrix:")
print(np.array2string(cm, separator=', '))

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
}

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(x_train, y_train)

# Print best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


knn = KNeighborsClassifier(n_neighbors=7)# Train the model
knn.fit(x_train, y_train)
# Test the model
predictions = knn.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"KNeighbors Accuracy: {accuracy}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)
print(f"KNeighbors Confusion Matrix:")
print(np.array2string(cm, separator=', '))

from sklearn.ensemble import VotingClassifier
# Initialize a VotingClassifier with soft voting
voting_clf = VotingClassifier(estimators=[('sfc', svc), ('rfc', rfc), ('mnb', mnb), ('knn', knn)], voting='hard')

# Fit the VotingClassifier
voting_clf.fit(x_train, y_train)

# Make predictions
y_pred = voting_clf.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")

import pickle
pickle.dump(voting_clf, open('dis_symp.pkl','wb'))

true_specialists = []
predicted_specialists = []

# Look up the specialist for each true and predicted disease
for disease in y_test:
    specialist = data.loc[data['Disease'] == disease, 'Specialist'].iloc[0]
    true_specialists.append(specialist)

for disease in y_pred:
    specialist = data.loc[data['Disease'] == disease, 'Specialist'].iloc[0]
    predicted_specialists.append(specialist)

# Calculate accuracy
correct_predictions = sum(1 for true, pred in zip(true_specialists, predicted_specialists) if true == pred)
accuracy = correct_predictions / len(y_test)

print("Accuracy:", accuracy)


symp_list = gen_data.columns.tolist()
symp_list.remove('Disease')

def encode_symptoms(symptoms, symptom_columns = symp_list):
    encoded_symptoms = [1 if column.lower().strip() in [symptom.lower().strip() for symptom in symptoms] else 0 for column in symptom_columns]
    return encoded_symptoms

def predict_disease(symps):
  enc_symps = encode_symptoms( symps)
  predicted_disease = voting_clf.predict([enc_symps])[0]
#   predicted_specialist = model.predict([enc_symps])[0]
  return  predicted_disease

