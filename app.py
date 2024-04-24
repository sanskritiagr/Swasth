import streamlit as st
import pickle
import pandas as pd
from datetime import time

import disease_predict
import specialist_predict
import doctor_rating
import suggest_doctors


gen_data = pd.read_csv('./gen_data.csv')

symp_list = gen_data.columns.tolist()
symp_list.remove('Disease')


# Define the Streamlit app
def app():
    st.title("Disease Specialist Prediction")

    # Get user symptoms from dropdown menu
    symptoms = st.multiselect("Select your symptoms", symp_list)

    if symptoms:
        dis = disease_predict.predict_disease(symptoms)
        spec = specialist_predict.predict(symptoms)

        # Display the predicted specialist
        st.write(f"Based on your symptoms, you have **{dis}**. So, you should consult a **{spec}**.")

        st.subheader(f"Please login to get personalized recommendation:")
        st.write("(If you are a New User, click 'New User')")
        user_id = st.number_input("Enter your user ID (1-1000):", min_value=1, max_value=1000, step=1)
        password = st.text_input("Enter your password(123):", type="password")

        ratings_docs = doctor_rating.avg_rating()

        if st.button("Submit"):
            ratings_docs = doctor_rating.col_filtering(user_id)
            ####run a function which takes user_id
        if st.button("New User"):
            ratings_docs = ratings_docs
            ###take mean
        
        ratings_doc = doctor_rating.merge_tables(ratings_docs)
        doc_list = suggest_doctors.find_doctors_by_specialty(spec)
        final_table = suggest_doctors.select_doctors_by_list(doc_list, ratings_doc)
        st.title(f"{spec}s in your area: ")
        # print(final_table.columns)
        selected_columns = ['Doctor Name', 'Rating', 'Telephone No.', 'Hospital Name', 'Opening Time', 'Closing Time']
        selected = final_table[selected_columns]
        st.dataframe(selected)




    else:
        st.write("Please select at least one symptom.")

# Run the Streamlit app
if __name__ == "__main__":
    app()
