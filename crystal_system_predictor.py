import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import chemparse
import streamlit as st

data = pd.read_csv("crystal_data.csv",delimiter = ',')

st.write('''
# Crystal System Prediction
 Integrating Concepts Of Chemistry & Computer Science 
''')

chem_data = data.Formula.apply(chemparse.parse_formula)
chem_data = pd.json_normalize(chem_data)
chem_data = chem_data.fillna(0)

data = data.join(chem_data)

st.subheader("Data Information")
st.dataframe(data)
st.write(data.describe())

user_input = st.sidebar.text_input("Enter The Formula Of The Compound")
user_input = chemparse.parse_formula(user_input)

element_list = ['K','S','O','Al','Fe','H','N','Ce','C','Cl','B','Cu','Ba','Ca','Co','Pb','Mn','Mg','Hg','Ni','Cr','Sr','Na','Zn','Ag','I','P']

user_data = {}

melting_point = {"Melting Point" : int(st.sidebar.slider("Melting Point (in K)",273.0,2000.0,664.1))}
solubility = {"Solubility" : int(st.sidebar.slider("Solubility In Water (in g/L)",0.0,5550.0,647.5))}
user_data.update(melting_point)
user_data.update(solubility)
for i in element_list:
    try:
        element = {i:int(user_input[i])}
        user_data.update(element)
    except KeyError:
        element_nil = {i:0}
        user_data.update(element_nil)

features = pd.DataFrame(user_data, index = [0])

x = data['Crystal_System']
y = data.drop(columns = ['Formula', 'Crystal_System'])

y_train,y_test,x_train,x_test = train_test_split(x,y, test_size = 0.2)

model = RandomForestClassifier()
model.fit(x_train,y_train)

if st.sidebar.button('Submit'):
    st.subheader("Predictions")
    predictions = model.predict(features)
    st.write(predictions)
else:
    print()
