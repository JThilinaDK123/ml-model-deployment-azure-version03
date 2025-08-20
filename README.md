
# Deploy LSTM Model in Azure

In this project, I will build and deploy a LSTM model using the crude oil price dataset to forecast crude oil price for a given time window.

The primary objective of this project is to create a web application that allows users to input time window details and get forecasted crude oil price values in real time. I will implement the application using Flask, a lightweight Python web framework, and deploy it on Microsoft Azure as a cloud-based service.

Once deployed, the web application will have a publicly accessible URL, providing an interactive interface where users can input data and get forecasted values instantly. This project demonstrates end-to-end machine learning workflow — from data preprocessing and model training to cloud deployment and web integration.



#### Note:
if youre using a linux os, apply below line under the configuration tab as a startup command

gunicorn --bind=0.0.0.0:$PORT app:app
