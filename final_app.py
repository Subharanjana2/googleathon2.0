import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import streamlit as st
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import cv2
from deepface import DeepFace
from pip import main
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'pain']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform real-time emotion detection
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        detected_emotion = emotion_labels[emotion_idx]
        return detected_emotion

# Streamlit app
st.set_page_config(page_title='AI Driven Music System', layout="wide")
st.title('AI Driven Music System ')
st.title("1.Emotion Analysis")
st.info("This interface utilizes AI for real-time emotion detection. The tracks shown are tailored based on detected facial expressions.")
# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Unable to open the webcam.")
    st.stop()

# Initialize valence with a default value
valence = 0.5

# Checkbox to start and stop face detection
start_detection = st.checkbox("Start Face Detection")

while start_detection:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform real-time emotion detection
    detected_emotion = detect_emotion(frame)

    # Update valence based on detected emotion
    if detected_emotion in ['happy', 'surprise']:
        valence += 0.1
    elif detected_emotion in ['angry', 'sad']:
        valence -= 0.1

    # Display the webcam feed with emotion information
    st.image(frame, channels="BGR")
    st.text(f"Detected Emotion: {detected_emotion}")
    st.text(f"Updated Valence: {valence:.2f}")

    # Function to recommend music based on emotion
    def recommend_music(emotion):
    # Your music recommendation logic goes here
    # You can use external APIs, databases, or pre-defined playlists to recommend music

    # Example: A simple mapping of emotions to music genres
            music_genres = {
           'angry': 'Metal',
           'disgust': 'Classical',
           'fear': 'Ambient',
            'happy': 'Pop',
            'sad': 'Blues',
            'surprise': 'EDM',
            'neutral': 'Instrumental',
            'pain': 'Rock'
           }

            recommended_genre = music_genres.get(emotion, 'Unknown')
            return recommended_genre
    genre = st.write(recommend_music(detected_emotion))
    break
    # Optional: Use the valence value to fetch music recommendations (replace this with your recommendation logic)
    # recommended_tracks = sp.recommendations(seed_tracks=[track_id], limit=5, target_audio_features={'valence': valence})

# Release the video capture object
cap.release()
cv2.destroyAllWindows()



# Spotify API credentials
client_id = '1138a66b81e2490593ab49b4573a0c5f'
client_secret = 'c7d18d1f37c44bd891128790a2f1b982'




# Initialize the Spotipy client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)




# Streamlit app
st.title('2.Suggesting Tracks From Spotify')
st.info("Based On Your Current Mood, suggest some Music that you would Like to hear so that it makes the recommendations much more relatable. Considering your current mood, could you recommend some music that you personally enjoy ?.  This would make the recommendations more relatable and tailored to your preferences.")



# Create columns for layout
col1, col2, col3 = st.columns([1, 3, 1])

# Initialize the selected_track variable
selected_track = None

selected_track_details = None



# Section for track selection
with col2:
    st.header('Search For a Track')
    
    # Input field for track search
    search_query = st.text_input('Select From the Available Tracks Based on Search')

    if search_query:
        # Search for tracks based on user input
        results = sp.search(q=search_query, type='track', limit=10)

        if results and results['tracks']['items']:
            st.subheader('Search Results')
            selected_track = st.selectbox('Select a track', [f"{track['name']} by {track['artists'][0]['name']}" for track in results['tracks']['items']])

            if selected_track:
                selected_track_details = results['tracks']['items'][[f"{track['name']} by {track['artists'][0]['name']}" for track in results['tracks']['items']].index(selected_track)]
                track_name = selected_track_details['name']
                artist_name = selected_track_details['artists'][0]['name']
                album_name = selected_track_details['album']['name']
                cover_url = selected_track_details['album']['images'][0]['url']
                track_id = selected_track_details['id']

                # Display the selected track details
                st.subheader('Selected Track Details')
                st.write(f"Track: {track_name}")
                st.write(f"Artist: {artist_name}")
                st.write(f"Album: {album_name}")
                
                st.image(cover_url,width = 200)

# Section for track analysis
# Section for track analysis
with col2:
    st.title('2.1.Track Analysis')
    
    if selected_track:
        selected_track_details = results['tracks']['items'][[f"{track['name']} by {track['artists'][0]['name']}" for track in results['tracks']['items']].index(selected_track)]
        track_uri = f'spotify:track:{selected_track_details["id"]}'
        
        
        
        # Load the full audio data for the selected track
        audio_url = sp.track(track_uri)['preview_url']

        
        st.info("You can preview the first 30 seconds of the song here. Please note that the full song cannot be played due to copyright issues.")

        

        # Display an audio player for the preview URL
        st.audio(audio_url, format="audio/mp3")

        if audio_url:
            response = requests.get(audio_url)
            y, sr = librosa.load(io.BytesIO(response.content))

            # Visualize highs and lows using a spectrogram
            st.subheader('Highs and Lows')
            D = np.abs(librosa.stft(y))
            fig = px.imshow(librosa.amplitude_to_db(D, ref=np.max), aspect='auto')
            
            # Adjust the size of the graph and display it using st.plotly_chart
            st.plotly_chart(fig, use_container_width=True)
            st.info("The 'Highs and Lows' graph shows the distribution of high and low frequencies over time. Brighter areas indicate higher energy in the frequency bands.")
            st.write("Adjust the following audio features for better recommendations:")
            st.write("- Increase 'Energy' for tracks with brighter areas, indicating higher energy.")
            st.write("- Experiment with 'Instrumentalness' to balance vocal and instrumental elements.")
            # Visualize beat intensity
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            times = librosa.times_like(onset_env)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
            st.subheader('Beat Intensity')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=onset_env, mode='lines', name='Onset strength'))
            fig.add_vline(x=times[onset_frames], line_color='red', line_dash='dash', name='Onsets')
            fig.update_layout(xaxis_title='Time (s)', yaxis_title='Onset Strength')

            # Adjust the size of the graph and display it using st.plotly_chart
            st.plotly_chart(fig, use_container_width=True)
            st.info("The 'Beat Intensity' graph represents the rhythm and intensity changes in the music. Red vertical lines indicate the onsets of beats in the track.")
            st.write("Adjust the following audio features for better recommendations:")
            st.write("- Increase 'Danceability' for tracks with pronounced beat onsets.")
            st.write("- Experiment with 'Energy' to match the intensity of the beats.")
            # Visualize energy changes
            rms = librosa.feature.rms(y=y)
            times = librosa.times_like(rms)
            st.subheader('Energy Changes')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=rms[0], mode='lines', name='RMS Energy'))
            fig.update_layout(xaxis_title='Time (s)', yaxis_title='RMS Energy')

            # Adjust the size of the graph and display it using st.plotly_chart
            st.plotly_chart(fig, use_container_width=True)
            st.info("The 'Energy Changes' graph displays the root mean square energy (RMSE) of the audio. It shows how the energy in the music changes over time.")
            st.write("Adjust the following audio features for better recommendations:")
            st.write("- Experiment with 'Energy' to align with the dynamic changes in the music.")
            st.write("- Adjust 'Valence' for a more positive or negative emotional tone.")


            # New plot: Visualize Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            st.subheader('MFCCs (Mel-frequency cepstral coefficients)')
            fig = px.imshow(mfccs, aspect='auto', origin='lower')
            fig.update_layout(xaxis_title='Time (s)', yaxis_title='MFCCs')
            st.plotly_chart(fig, use_container_width=True)
            st.info("The 'MFCCs' graph visualizes the Mel-frequency cepstral coefficients, which represent the short-term power spectrum of a sound signal.")
            st.write("Adjust the following audio features for better recommendations:")
            st.write("- Experiment with 'Instrumentalness' to balance vocal and instrumental elements.")
            st.write("- Consider 'Valence' to match the emotional tone represented by the MFCCs.")
            
        else:
            st.warning("Audio analysis not available for the selected track.")

# Streamlit app
st.title('3. Music Recommendation Based on Custom Audio Features')

# Create sliders for adjusting audio features
st.header('Adjust Audio Features')

acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5, 0.01)
st.info("Acousticness measures the amount of acoustic sound in the track. Decreasing it may lead to more electronic or non-acoustic recommendations, while increasing it may result in more acoustic recommendations.")
danceability = st.slider('Danceability', 0.0, 1.0, 0.5, 0.01)
st.info("Danceability quantifies how suitable the track is for dancing. Higher values represent tracks that are more danceable.")
energy = st.slider('Energy', 0.0, 1.0, 0.5, 0.01)
st.info("Energy measures the intensity and activity in the music. Higher values indicate more energetic tracks, while lower values represent calmer tracks.")
instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.5, 0.01)
st.info("Instrumentalness assesses the presence of vocals in the track. Lower values suggest the presence of vocals, while higher values indicate instrumental tracks.")
liveness = st.slider('Liveness', 0.0, 1.0, 0.5, 0.01)
st.info("Liveness indicates the likelihood of a live audience in the track. Higher values suggest a live performance, while lower values imply a studio recording.")
speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5, 0.01)
st.info("Speechiness quantifies the presence of spoken words in the track. Higher values indicate more speech-like audio, while lower values are more musical and instrumental.")
# Slider to manually adjust valence (if needed)
st.text(f"Updated Valence: {valence:.2f}")
valence = st.slider('Valence', 0.0, 1.0, 0.5, 0.01)
st.info("Valence measures the overall positivity of the track. Higher values represent happier and more positive tracks, while lower values indicate sadder or more negative tracks.                                                                                                                                                                                 Please Keep the Updated Valence Recommended By the System to Achieve Better Recommendations")

# Button to generate recommendations
if st.button('Generate Similar Tracks'):
    # Build the target_audio_features dictionary
    target_audio_features = {
        'acousticness': acousticness,
        'danceability': danceability,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'speechiness': speechiness,
        'valence': valence
    }

     # Fetch track recommendations based on the selected audio features
    recommended_tracks = sp.recommendations(seed_tracks=[track_id], limit=5, target_audio_features=target_audio_features,seed_genres=genre)

    st.subheader('Recommended Tracks:')
    
    if selected_track:
        # Fetch track recommendations based on the selected audio features
        recommended_tracks = sp.recommendations(seed_tracks=[track_id], limit=5, target_audio_features=target_audio_features,seed_genres=genre)

    # Create columns for the recommended tracks, e.g., 2 tracks per column
    num_columns = 2
    num_recommended_tracks = len(recommended_tracks['tracks'])
    num_rows = (num_recommended_tracks + num_columns - 1) // num_columns
    
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col in range(num_columns):
            idx = row * num_columns + col
            if idx < num_recommended_tracks:
                rec_track = recommended_tracks['tracks'][idx]
                with cols[col]:
                    st.write(f"Track: {rec_track['name']}")
                    st.write(f"Artist: {rec_track['artists'][0]['name']}")
                    st.write(f"Album: {rec_track['album']['name']}")
                    album_art_url = rec_track['album']['images'][0]['url']
                    
                    st.write(f"Original URL: [Listen on Spotify]({rec_track['external_urls']['spotify']})")
                    st.image(album_art_url,width=250)

                    # Get the 'preview_url' for the recommended track
                    recommended_audio_url = rec_track['preview_url']

                    # Display an audio player for the recommended track's preview URL
                    st.audio(recommended_audio_url, format="audio/mp3")

    
    
    st.subheader('A Few More Tracks Based On Mood:')
    st.info("Considering the detected mood through facial emotion analysis, the music genre that would best suit you is . Based on this suggested genre, we generate a selection of tracks that align with it.")

    if selected_track:
        recommendedm_tracks = sp.recommendations(seed_tracks=[track_id], limit=2, target_audio_features=target_audio_features, seed_genres=genre)

# Calculate the number of columns
        numm_columns = 2
        num_recommendedm_tracks = len(recommendedm_tracks['tracks'])
        numm_rows = (num_recommendedm_tracks + numm_columns - 1) // numm_columns

# Use st.columns with width parameter for control
        colsm = st.columns(numm_columns)

        for row in range(numm_rows):
            for colm in range(numm_columns):
                idxm = row * numm_columns + colm
                if idxm < num_recommendedm_tracks:
                    recm_track = recommendedm_tracks['tracks'][idxm]
                with colsm[colm]:
                    st.write(f"Track: {recm_track['name']}")
                    st.write(f"Artist: {recm_track['artists'][0]['name']}")
                    st.write(f"Album: {recm_track['album']['name']}")
                    album_artm_url = recm_track['album']['images'][0]['url']
                    st.write(f"Original URL: [Listen on Spotify]({recm_track['external_urls']['spotify']})")
                    st.image(album_artm_url, width=250)
                    st.write(f"Original URL: [Listen on Spotify]({recm_track['external_urls']['spotify']})")
                    recommendedm_audio_url = recm_track['preview_url']
                    st.audio(recommendedm_audio_url, format="audio/mp3")
