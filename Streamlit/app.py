import streamlit as st
from PIL import Image
from clf import predict_image
file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
	image = Image.open(file_up)
	st.image(image, caption='Uploaded Image.', use_column_width=True)
	# create a button to make predictions on the image
	if st.button('Predict'):
		label = predict_image(image)
		if label == 0:
			label = "Fractured"
		else:
			label = "Not Fractured"
		st.write("The image is:", label)
else:
	st.write("Please upload an image file.")