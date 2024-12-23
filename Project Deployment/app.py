# -*- coding: utf-8 -*-
"""APP

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HCtA_V4HnbMc5XtOywCF0oDLmZsAAsQq
"""

import pickle
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# animation
def load_lottieurl(url):
     r = requests.get(url)
     if r.status_code !=200:
          return None
     return r.json()

lottie_cod = load_lottieurl("https://lottie.host/16afc6aa-adad-4629-93df-955ecba91f3a/ZQnoqtt3pT.json")

model = "final.pkl"
with open(model,"rb") as file:
  final = pickle.load(file)

cv = "cv.pkl"
with open(cv,"rb") as file:
  cv = pickle.load(file)

def preprocess_val(val):
    return val

st.title(":chart_with_upwards_trend: Sentiment Analysis Prediction Using Logistic Regression")

def local_css(file_name):
  with open(file_name)as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

local_css("style.css")

animation_symbol = "❄️"

st.markdown(f"""
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>
<div class="snowflake">{animation_symbol}</div>""",unsafe_allow_html=True)

# defining emojis
emoji_positive = "😃"
emoji_negative = "😞"
emoji_neutral = "😐"

#creating glowing colour effect with css
def get_glowing_css(color):
  return f"""@keyframes glowing{{
    0%{{
      text-shadow: 0 0 5px{color};
    }}
    50%{{
      text-shadow: 0 0 20px{color};
    }}
    100%{{
      text-shadow: 0 0 5px{color};
    }}
  }}"""

#css for glowing effect
glowing_css = get_glowing_css("#ffd700")

st.markdown(f"<style>{glowing_css}</style>",unsafe_allow_html=True)

user_input = st.text_area("Enter the Valid Sentence for Sentiment Prediction","")
preprocessed_input = preprocess_val(user_input)

# making prediction

if st.button("predict"):
  if not user_input or not cv.transform([preprocessed_input]).nnz:
    st.warning("Please Enter the Valid Sentence For Prediction")
  else:
    try:
      user_input = cv.transform([preprocessed_input])
      prediction = final.predict(user_input)
      senti_labels = ["Negative","Neutral","Positive"]
      prediction_sentiment = senti_labels[prediction[0]]
      #displaying sentiment prediction
      st.write(f"Predicted Sentiment: {prediction_sentiment}")
      if prediction[0]== 0:
        glowing_emoji = emoji_negative
        glowing_colour = "#ffd700"
      elif prediction[0]== 1:
        glowing_emoji = emoji_neutral
        glowing_colour = "#cOcOcO"

      elif prediction[0] == 2:
        glowing_emoji = emoji_positive
        glowing_colour = "#ff6347"
      st.markdown(f'<span style="font-size: 4em; animation: glowing 1.5s infinite; color: {glowing_colour};">{glowing_emoji}</span>',
                        unsafe_allow_html=True)




    except Exception as e:
      st.error(f"An Error Occured: {e}")

st_lottie(lottie_cod,height = 300,key = "coding")
