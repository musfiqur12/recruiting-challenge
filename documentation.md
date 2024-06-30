# Facial Profile Creator Documentation

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Facial Analysis](#facial-analysis)
- [Facial Recognition](#facial-recognition)
- [Utilizing Facial Recognition for Employee Attendace and Verification](#utilizing-facial-recognition-for-employee-attendace-and-verification)
- [Improving the Application](#improving-the-application)

## Overview

The Facial Profile Creator is a FastAPI application that allows users to upload an image and generates a "facial profile" that utilizes Dlib's pre-trained models and methods.  

## Technologies
 
The Facial Profile Creator was developed using the following technologies:
- [FastAPI]()
- [OpenCV]()
- [Dlib]()

FastAPI is utilized to allow users to upload images to a basic  server. OpenCV is used to preprocess the uploaded images while the Dlib library is used to detect faces and perform facial analysis and recognition. 

## Facial Analysis

To conduct facial analysis, we first preprocess the image using OpenCV. The image is first decoded and converted to grayscale. Then, we detect the face in the image using Dlib's get_frontal_face_detector method and use Dlib's shape_predictor method to extract facial landmarks. 

These landmarks include:
- Mouth
- Nose
- Jawline
- Eyes 
- Eyebrows

The landmark extractor utilizes the pixel intensities to determine the position of 68 points on the face that map to facial structures.  

![Facial Landmarks](https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg)

Once the data is extracted, it is stored in a that dictionary maps a unique id to a class called Profile. This class stores the name associated with the unique id, a list of data extracted from provided images, and an element-wise average of the data from each image. 

Dlib also has a 5-point facial landmark detector as well as a 198-point facial landmark detector. I chose to utilize Dlib's 68-point facial landmark detector due to the accuracy and performance provided. It also allows for [real-time facial detection and extraction of features](http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html) which could be useful in use cases relevant to IdentifAI. 


## Facial Recognition

To conduct facial recognition, we first perform analysis on the uploaded image to create a facial profile. Then, we compute the Euclidian norm to calculate the distance between the uploaded image and the averaged data for each of our profiles. If the distance is under a certain threshold, we return the name of that profile. 

## Utilizing Facial Recognition for Employee Attendace and Verification

The facial profiles generated from the analysis can be utilized to check employee attendance on a zoom call. The application can be modified to detect multiple faces in one screenshot and perform analysis on each of those faces to generate a facial profile. Those profiles can then be compared with existing data to detect whether an employee is in the meeting. 

Another use of the facial profiles is ensuring that the employees in attendance are actually who they say they are. We could also ensure that virtual meetings don't have any uninvited attendees. The process would be very similar to checking employee attendance. However, we would have to compare the name returned from the facial recognition with the name provided when they sign in. 

## Improving the Application

There are several things that can be done to improve the application including:
- Performing validation testing to find an optimal threshold
- Exploring other algorithms for comparison such as cosine similarity which tends to perform better for higher-dimensional comparisons
- Exploring models such as FaceNet which utilizes triplet loss, DeepFace, ArcFace