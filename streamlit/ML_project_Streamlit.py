import streamlit as st
from PIL import Image
from image_classification import plant_disease_detection
from image_classification import find_the_remedy
import requests
import json
import time
from datetime import datetime

st.set_page_config(layout="wide")

col1, col2 = st.columns(2,gap="large")

with col1:
    st.title("Plant Disease Image Classification")
    st.header("Find the Disease your Plant suffers from with just a click!")
    
    uploaded_file = st.file_uploader("Upload a plant image ...", type="jpg")
    
    def time_format_for_location(utc_with_tz):
        local_time = datetime.utcfromtimestamp(utc_with_tz)
        return local_time.time()
 
    def showWeather(city):

        if city != "":
            #Enter you api key, copies from the OpenWeatherMap dashboard
            api_key = "f3d5317ed9f51f576fdcb096c17fca96"  #sample API
 
            # API url
            weather_url = 'http://api.openweathermap.org/data/2.5/weather?q=' + city + '&appid='+api_key
 
            # Get the response from fetched url
            response = requests.get(weather_url)
 
            # changing response from json to python readable 
            weather_info = response.json()
     
            #as per API documentation, if the cod is 200, it means that weather data was successfully fetched
 
            if weather_info['cod'] == 200:
                kelvin = 273 # value of kelvin
 
                #-----------Storing the fetched values of weather of a city
 
                temp = int(weather_info['main']['temp'] - kelvin)  #converting default kelvin value to Celcius
                feels_like_temp = int(weather_info['main']['feels_like'] - kelvin)
                pressure = weather_info['main']['pressure']
                humidity = weather_info['main']['humidity']
                wind_speed = weather_info['wind']['speed'] * 3.6
                sunrise = weather_info['sys']['sunrise']
                sunset = weather_info['sys']['sunset']
                timezone = weather_info['timezone']
                cloudy = weather_info['clouds']['all']
                description = weather_info['weather'][0]['description']
 
                sunrise_time = time_format_for_location(sunrise + timezone)
                sunset_time = time_format_for_location(sunset + timezone)
 
                #assigning Values to our weather variable, to display as output
         
                weather = (f"Weather of: {city} \t\nTemperature (Celsius): {temp}° \nFeels like in (Celsius): {feels_like_temp}° \nPressure: {pressure} hPa \t\nHumidity: {humidity}%\nSunrise at {sunrise_time} and Sunset at {sunset_time}\nCloud: {cloudy}%\nInfo: {description}",)
            
     
            else:
                weather = (f"\n\tWeather for '{city}' not found!\n\tKindly Enter valid City Name !!",)
 
        else:
            weather = ("\n\tNo city yet",)
        return weather

    st.header("Check the weather at your location!")
    city = st.text_input('Enter city here', "")
    weather = showWeather(city)
    
    #st.markdown('<p style="font-family:Arial; color:white; font-size: 35px;">%s</p>' % city, unsafe_allow_html=True)
    st.markdown('<p style="font-family:Arial; color:white; font-size: 25px;">%s</p>' % weather[0], unsafe_allow_html=True)
    

with col2:
    st.write("")
    st.write("")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded plant picture', use_column_width=False)
        st.write("")
        with st.spinner(text= 'Classifying...'):
            label = plant_disease_detection(image, 'model_VGG19_2.h5')
            time.sleep(1)
            st.success('Successful Image Classification!')
        disease,remedy = find_the_remedy(label)
        st.markdown('<p style="font-family:Arial; color:red; font-size: 20px;">Your plant disease is:</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Arial; color:#d4aa02; font-size: 25px;">%s</p>' % disease, unsafe_allow_html=True)
        st.markdown('<p style="font-family:Arial; color:red; font-size: 20px;">Recommended Remedy:</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Arial; color:#d4aa02; font-size: 25px;">%s</p>' % remedy, unsafe_allow_html=True)
        

