import torch
from model import LocalizationModel
import streamlit as st
from preprocessing import preprocess
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

@st.cache_resource()
def load_model():
    model = LocalizationModel()
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    return model

model = load_model()

def predict(img):
    img = preprocess(img)
    with torch.no_grad():
        pred, box = model(img)
    return pred, box

st.title('Draw number get prediction')

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    update_streamlit=True,
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')

        pred, box = predict(img)
        
        st.write(f"Prediction: {pred.argmax()}")
        
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        box = box.cpu().numpy()[0]
        box = (box * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).astype(int)
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
        
        st.image(img)