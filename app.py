import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Song Mood Predictor", page_icon=":headphones:", layout="wide")
# Function to load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


col1, col2 = st.columns([1,4])
with col1:
    # Load Lottie animation for the app icon
    lottie_url = "https://lottie.host/58ab2799-0df1-427e-a744-1d081f92b15a/HQluYx5o8z.json"
    lottie_json = load_lottieurl(lottie_url)
    app_icon = st_lottie(lottie_json, speed=1, width=100, height=100, key="lottie")
with col2:
# Initial welcome message
    welcome_container = st.empty()
    welcome_container.header("Spotify Mood Recommender System")


    st.write("Enter the Values of the Song that you want, the model will classify the mood of song")


# Load the trained model
model_path = 'music_mood_model.pkl'
with open(model_path, 'rb') as model_file:
   model = pickle.load(model_file)

# Mapping of moods to emojis
mood_to_emoji = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'nostalgia': '🌟',
    'calm': '😌',
}

def get_numeric_input(prompt):
    user_input = st.text_input(prompt)

    # Check if the input is not empty and is a valid numeric value
    if user_input and user_input.replace('.', '', 1).isdigit():
        return float(user_input)
    else:
        print(f"Warning: Invalid numeric value entered for {prompt.replace('(numeric value)', '').strip()}.")
        return None

# Number inputs for dance, energy, loudness, valence, tempo
dance = st.number_input("Enter Dance (numeric value):", value=None, step=None, format="%.2f")
energy = st.number_input("Enter Energy (numeric value):", value=None, step=None, format="%.2f")
loudness = st.number_input("Enter Loudness (numeric value):", value=None, step=None, format="%.2f")
valence = st.number_input("Enter Valence (numeric value):", value=None, step=None, format="%.2f")
tempo = st.number_input("Enter Tempo (numeric value):", value=None, step=None, format="%.2f")

# Two input modes (True or False)
input_mode = st.radio("Select Input Mode:", [True, False])

# Encode input_mode as 0 or 1
input_mode_encoded = 1 if input_mode else 0

genres = ['bollywood', 'classical', 'folk', 'ghazal', 'indie', 'pop', 'qawwali', 'reggae', 'rock', 'sufi']
# Genre selection using radio buttons
selected_genre = st.radio("Select Genre:", genres)

# Encode the selected genre
genre_mapping = {genre: 1.0 if genre == selected_genre else 0.0 for genre in genres}

# Range sliders
instrumentalness = st.slider("Instrumentalness:", 0.0, 3.0, step=1.0)
speechiness = st.slider("Speechiness:", 0.0, 3.0, step=1.0)
acousticness = st.slider("Acousticness:", 0.0, 4.0, step=1.0)
liveness = st.slider("Liveness:", 0.0, 4.0, step=1.0)

st.markdown("<br>", unsafe_allow_html=True)
# Button to trigger recommendations or further processing
if st.button("Predict Mood"):

    # Prepare the user input for the model
    user_input = pd.DataFrame({
        'dance': [dance],
        'energy': [energy],
        'loudness': [loudness],
        'valence': [valence],
        'tempo': [tempo],
        'input_mode': [input_mode_encoded],
        **genre_mapping,
        'instrumentalness': [instrumentalness],
        'speechiness': [speechiness],
        'acousticness': [acousticness],
        'liveness': [liveness],
    })

    st.markdown("<br>", unsafe_allow_html=True)

    # Make predictions using the model
    predicted_mood = model.predict(user_input)

    # Convert NumPy array to string by accessing the first element
    predicted_mood_str = str(predicted_mood[0]) if isinstance(predicted_mood, np.ndarray) else str(predicted_mood)

    # Fetch the corresponding emoji for the predicted mood
    predicted_emoji = mood_to_emoji.get(predicted_mood_str.lower(), '🤔')

    st.success(f"Predicted Mood: {predicted_mood_str} {predicted_emoji}")


st.markdown(
    """
    <style>
        .footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: #1e1e1e; /* Dark color */
        }
        .footer p {
            margin: 0;
            color: #fff; /* Text color */
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            text-decoration: none;
            color: #fff; /* Link color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        <p>Made By <b>White Hat Team</b></p>
        <a href="https://github.com/Geeky-Hassan/Spotify-Mood-Recommender-System/" target="_blank">
            <img src="https://media.istockphoto.com/id/511653492/vector/incognito-hacker-spy-agent.jpg?s=612x612&w=0&k=20&c=IO9KG36TQ2wnIR3idSk-oEDV9zx5BsyduKQAlWPhNJU=" alt="White Hat Logo" width="30">
            Github Repository
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

