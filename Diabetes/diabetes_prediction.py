import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from random import randint


st.title('Diabetes Prediction')
st.subheader('By Pyae Sone Phyo Thu')

data = pd.read_csv('diabetes.csv')

non_diabetic=data[data.Outcome==0]
diabetic=data[data.Outcome==1]

#non_diabetic_sample=non_diabetic.sample(n=268)
#diabetes=pd.concat([non_diabetic_sample,diabetic],axis=0)
#non_diabetic_sample=non_diabetic.sample(n=268)


data_check = st.sidebar.checkbox('Diabetes data')

metrics_check=st.sidebar.checkbox('Metrics Check')

if data_check:
	st.write(data)
	#st.write(diabetes.shape)

def user_input_features():

	col1,col2=st.columns(2)
	with col1:
		Pregnancies=st.number_input('Number of Pregnancies',min_value=0, max_value=17, value=0, step=1)
	with col2:
		Glucose=st.number_input('Glucose Level',min_value=0, max_value=199, value=0, step=1)

	col1,col2=st.columns(2)
	with col1:
		BloodPressure=st.number_input('Blood Pressure Value',min_value=0, max_value=122, value=0, step=1)
	with col2:
		SkinThickness=st.number_input('Skin Thickness Value',min_value=0, max_value=99, value=0, step=1)

	col1,col2=st.columns(2)
	with col1:
		Insulin=st.number_input('Insulin Value',min_value=0, max_value=846, value=0, step=1)
	with col2:
		BMI=st.number_input('BMI Value',min_value=0.0,max_value=67.1,value=33.3)

	col1,col2=st.columns(2)
	with col1:
		DiabetesPedigreeFunction=st.number_input('Diabetes Pedigree Function Value',min_value=0.078,max_value=2.42,value= 1.20)
	with col2:
		Age=st.number_input('Age:',min_value=1, max_value=81, value=1, step=1)
	
	data={'Pregnancies':Pregnancies,
			'Glucose':Glucose,
			'BloodPressure':BloodPressure,
			'SkinThickness':SkinThickness,
			'Insulin':Insulin,
			'BMI':BMI,
			'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
			'Age':Age}
	features=pd.DataFrame(data,index=[0])#starting number in df
	return features


X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=7)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

st.subheader('Using Linear Regression')

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train,y_train)

	#Evaluation
x_train_prediction=model.predict(x_train)
x_test_prediction=model.predict(x_test)

input_df=user_input_features()	
#st.write(input_df)

test_button=st.button("Test")

if test_button:

	input_data_as_numpy_array=np.asarray(input_df)
	input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
		#mm_data=mm_scaler.transform(input_data_reshaped)
	std_data=scaler.transform(input_data_reshaped)

	prediction=model.predict(std_data)
	if prediction[0]==0:
		st.success('This person is NON Diabetic')
	if prediction[0]==1:
		st.success('This person is Diabetic')


if metrics_check:
	st.subheader('\nFor Standardized testing data')
	col1,col2=st.columns([2,1])
	with col1:
		st.info('Accuracy Score:')
	with col2:
		st.info(accuracy_score(y_test, x_test_prediction))

	with col1:
		st.info('Precision Macro Score:')
	with col2:
		st.info(precision_score(y_test, x_test_prediction,average = 'macro'))

	with col1:
		st.info('Recall Score:')
	with col2:
		st.info(recall_score(y_test, x_test_prediction, average = 'macro'))

	with col1:
		st.info('F1 Score:')
	with col2:
		st.info(f1_score(y_test, x_test_prediction, average = 'macro'))

	#with col1:
	#	st.info('Confusion Matrix:')
	#with col2:
	#	st.info(confusion_matrix(y_test,x_test_prediction))

	cm=confusion_matrix(y_test,x_test_prediction)
	st.write('\nConfusion Matrix:')
	st.write(cm)
