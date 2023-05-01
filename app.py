
import cv2
import nltk
import pytesseract
import numpy as np
from statistics import mode
from collections import Counter
from keras.models import load_model
from keras.preprocessing import image
from nltk.metrics.distance import jaccard_distance

import streamlit as st
# from pyngrok import ngrok 
# import pickle
from PIL import Image

st.set_page_config(
    page_title=""
)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Cognitive Impairment')
uploaded_file = st.file_uploader('Upload an image...', type=["JPEG", "JPG", "PNG"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    pix = np.array(img)

    print(pix)
    # st.image(img, caption='Uploaded Image')

    if st.button('Predict'):
        with st.spinner('Processing...'):
            
            _, img_encoded = cv2.imencode('.jpg', pix)

            # Decode the JPEG-encoded image string
            img1 = cv2.imdecode(img_encoded, cv2.IMREAD_GRAYSCALE)
            # img1 = cv2.imdecode(pix, cv2.IMREAD_GRAYSCALE)
            reference_text = "One day a zebra found a xylophone on the sidewalk. He quickly ran over, picked it up, and gave it to his pet mule. Just then he found another xylophone. He kept that one for himself"

            # Pass the image to the OCR engine
            text = pytesseract.image_to_string(img1)

            ocr_text = text
            ocr_tokens = nltk.word_tokenize(ocr_text)
            ocr_tokens = list(ocr_text)
            reference_tokens = list(reference_text)

            # Calculate the Jaccard similarity
            similarity = 1 - nltk.jaccard_distance(set(ocr_tokens), set(reference_tokens))

            if similarity>0.68:
                st.write("The Child Is Not Dysgraphic and has no Issues")
            else:
                model3 = load_model('model.h5')
                model4 = load_model('cnn3model.h5')

                # Load the image and invert it
                image = cv2.bitwise_not(img1)

                # Apply Otsu thresholding
                _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Find the contours of the thresholded image
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                predictions = []

                # Iterate through the contours and extract each character using bounding box
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    character = thresholded[y:y+h, x:x+w]
                    area = cv2.contourArea(contour)
                    if(area>10 and area<300):
                        character = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
                        character = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)
                        character = cv2.resize(character, (224,224), interpolation=cv2.INTER_AREA)
                        character = np.array(character)
                        character = np.expand_dims(character, axis=0)
                        character = character / 255.
                        predictions1 = model3.predict(character)
                        predictions2 = model4.predict(character)
                        
                        # # Assign weights to each model
                        weight1 = 0.45
                        weight2 = 0.55
                        final_predictions = []  
                        for i in range(len(predictions1)):
                            final_prediction = (predictions1[i] * weight1) + (predictions2[i] * weight2)
                            final_prediction = np.argmax(final_prediction, axis=-1)
                            final_predictions.append(final_prediction)

                        predictions.append(final_predictions[0])

                counts = Counter(predictions)

                # Print the count of each class
                for label, count in counts.items():
                    if label == 0:
                        st.write("Corrected:", count)
                    elif label == 1:
                        st.write("Normal:", count)
                    else:
                        st.write("Reversal:", count)

                reversal_count = counts.get(2, 0)
                normal_count = counts.get(1, 0)
                corrected_count = counts.get(0, 0)
                total = normal_count + corrected_count
                if total != 0:
                    ratio = reversal_count / total
                    if ratio>0.21:
                        st.write("")
                        st.write("The Child is Dsygraphic and Attention Might be required")
                    else:
                        st.write("The Child Is Not Dysgraphic")
                else:
                    st.write("The Child is Dsygraphic and Attention Might be required")
        st.success('DONE')