import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Moath/Documents/Machine-Learning/Regression/fitness-calorie-burn-rate-predict-99/trained_model.sav', 'rb'))

def calories_prediction(inputs):
    
    input_data_as_numpy_array = np.asarray(inputs)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return(prediction)

    
def main():
        st.title('needed calories calculator')
        
        User_ID = st.text_input('not matter, input any num')
        Gender = st.text_input('1 (male) , 0 (female)')
        Age = st.text_input('your age')
        Height = st.text_input('height')
        Weight = st.text_input('Weight')
        Duration = st.text_input('Duration')
        Heart_Rate = st.text_input('Heart_Rate')
        Body_Temp = st.text_input('Body_Temp')
        
        d = ''
        
        if st.button('needed to burn'):
            d = calories_prediction([User_ID, Gender, Age, Height, Weight, Duration,Heart_Rate, Body_Temp])
            
            st.success(d)
            
            
if __name__ == '__main__':
    main()