import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa

def findMax(arr):
    
    
    # Find the index of the maximum value
    max_index = np.argmax(arr)
    
    return max_index


def findGenre(ind):
    st.write("This is the index: ",ind)
    if ind == 0:
        return 'blues'
    elif ind == 1:
        return 'classical'
    elif ind == 2:
        return 'country'
    elif ind == 3:
        return 'disco'
    elif ind == 4:
        return 'hiphop'
    elif ind == 5:
        return 'jazz'
    elif ind == 6:
        return 'metal'
    elif ind == 7:
        return 'pop'
    elif ind == 8:
        return 'reggae'
    elif ind == 9:
        return 'rock'


# Function to extract audio features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y))
    harmony_mean = np.mean(librosa.effects.harmonic(y))
    harmony_var = np.var(librosa.effects.harmonic(y))
    perceptr_mean = np.mean(librosa.effects.percussive(y))
    perceptr_var = np.var(librosa.effects.percussive(y))
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = [np.mean(mfcc) for mfcc in mfccs]
    mfccs_var = [np.var(mfcc) for mfcc in mfccs]
    return [chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo] + mfccs_mean + mfccs_var

# Streamlit UI
st.header(":green[MUSIC GENRE CLASSIFIER]", divider="green")
st.subheader(":green[Predict Music Genre Through Audio Feature]")
st.markdown("#")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
model = tf.keras.models.load_model("genre_model.h5")
audio_features = None
if uploaded_file is not None:
    # Call feature extraction function
    audio_features = extract_features(uploaded_file)
    # st.write("Extracted Audio Features:")
    # st.write("Length of features:", len(audio_features))  # Print the length of the feature list
    # st.write("Features:", audio_features)


    # Display extracted features
    # st.write("Extracted Audio Features:")
    # feature_names = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo'] + [f'mfcc{i}_mean' for i in range(1, 21)] + [f'mfcc{i}_var' for i in range(1, 21)]
    # for feature_name, feature_value in zip(feature_names, audio_features):
    #     st.write(f"{feature_name}: {feature_value}")

    # Load the trained model
    

if st.button(':green[Classify Genre]'):
    # Prepare features to feed into the model
    features_to_feed = np.array(audio_features).reshape(1, -1)  # Reshape to a 2D array
    st.write("Feature to Feed")
    st.write(features_to_feed)


    # pr = np.array([0.35008812, 0.088756569, 0.130227923, 0.002826696, 1784.16585, 129774.0645, 2002.44906, 85882.76132, 3805.839606, 
    #            901505.4255, 0.083044821, 0.000766946, -4.53E-05, 0.008172282, 7.78E-06, 0.005698182, 123.046875, -113.5706482, 
    #            2564.20752, 121.5717926, 295.9138184, -19.16814232, 235.5744324, 42.36642075, 151.1068726, -6.364664078, 167.9347992, 
    #            18.62349892, 89.18083954, -13.7048912, 67.66049194, 15.34315014, 68.93257904, -12.27410984, 82.20420074, 10.97657204, 
    #            63.38631058, -8.326573372, 61.77309418, 8.803792, 51.24412537, -3.6723001, 41.21741486, 5.7479949, 40.55447769, 
    #            -5.162881851, 49.77542114, 0.752740204, 52.42090988, -1.690214634, 36.52407074, -0.408979177, 41.59710312, 
    #            -2.303522587, 55.06292343, 1.221290708, 46.93603516]).reshape(1, -1) 



    # Make prediction
    predict = model.predict(features_to_feed)
    st.write("Predict: ")
    st.write(predict)
    
    # Get the index of the maximum predicted value
    predicted_index = np.argmax(predict)
    
    # Map the index to the corresponding genre
    predicted_genre = findGenre(predicted_index)
    
    st.markdown("#")
    st.subheader(f'Predicted Genre: :green[{predicted_genre}]')








# 'blues',
#  'classical',
#  'country',
#  'disco',
#  'hiphop',
#  'jazz',

#  'metal',
#  'pop',
#  'reggae',
#  'rock'