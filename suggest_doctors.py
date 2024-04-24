import pandas as pd

doc_tb = pd.read_csv('./doc_table.csv')

def find_doctors_by_specialty(specialty):
    # Strip white spaces from the specialty column in the DataFrame
    doc_tb['Specialty'] = doc_tb['Specialty'].str.strip()

    # Strip white spaces from the input specialty
    specialty = specialty.strip()

    # Filter the DataFrame based on the given specialty (ignoring leading and trailing white spaces)
    filtered_df = doc_tb[doc_tb['Specialty'] == specialty]

    # Get the list of doctor names with the given specialty
    doctors_with_specialty = filtered_df['Doctor Name'].tolist()

    return doctors_with_specialty

def select_doctors_by_list(doctor_list, doc_tab):

    # Filter doctors based on the list

    selected_doctors_df = doc_tab[doc_tab['Doctor Name'].isin(doctor_list)]
    
    return selected_doctors_df
