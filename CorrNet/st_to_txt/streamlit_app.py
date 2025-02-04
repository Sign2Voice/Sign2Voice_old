import sys
import streamlit as st
import os
import shutil
from PIL import Image
import time  # Import for time measurement
import cv2

# 🛠 Add root directories to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath("Gloss2Text2Speech"))

# ✅ Import custom functions
from st_to_txt.process_frames import process_frames
from gloss_to_text import gloss_to_text
from text_to_speech import text_to_speech

# 📌 Define paths
UPLOAD_FOLDER = "st_to_txt/data/uploaded_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PHOENIX_MODEL_PATH = "./pretrained_model/dev_18.90_PHOENIX14.pt"
ADAPTER_MODEL_PATH = "./Gloss2Text2Speech/pretrained/adapter_model.bin"

# 🎬 **Streamlit App Title**
st.title("Talk to the Hand 👋🏽")

# --- **Upload Section** ---
st.title("📂 Upload Frames")
uploaded_files = st.file_uploader(
    "Upload your frames here (only images)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    st.write("🔄 Processing uploaded frames...")

    # 📂 Save uploaded frames
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.write("✅ Frames saved. Running model...")

    # 🧠 Run model to get glosses
    glosses = process_frames(PHOENIX_MODEL_PATH, UPLOAD_FOLDER, language="phoenix")

    if glosses:
        st.session_state.glosses = glosses  # 🌟 Store glosses
        st.subheader("📖 Predicted Glosses")
        st.write("📝", " ".join(glosses))

        # 🔄 **Automatically Generate Sentence & Speech**
        with st.spinner("⏳ Generating sentence and audio..."):
            gloss_text = " ".join(glosses)

            # 🧠 Convert Glosses → Sentence
            generated_sentence = gloss_to_text(gloss_text)
            st.subheader("📜 Generated Sentence")
            st.write("📢", generated_sentence)

            # 🔊 Convert Sentence → Speech
            text_to_speech(generated_sentence)
            st.success("✅ Text-to-Speech completed!")

# --- **Frame Preview** ---
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} frames uploaded successfully!")

    # 📂 **Clear the upload folder** (old frames)
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # 🖼 Save new uploaded frames
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        frame_path = os.path.join(UPLOAD_FOLDER, f"frame_{idx:03d}.jpg")
        image.save(frame_path)

    # 🖼 **Preview uploaded frames**
    st.subheader("📸 Frame Preview")
    image_files = sorted(os.listdir(UPLOAD_FOLDER))[:5]  # Show max. 5 images
    st.image([os.path.join(UPLOAD_FOLDER, img) for img in image_files], caption=image_files, width=200)





st.write("Or")


# Stream

st.write("🎥 Start a livestream here.")

# Status variable for the stream
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# Status variable for frame count
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0  # Initialize frame_count

# Status variable for the timer
if "start_time" not in st.session_state:
    st.session_state.start_time = None  # Start time for the timer

# Status variable for the last elapsed time
if "last_elapsed_time" not in st.session_state:
    st.session_state.last_elapsed_time = None  # Last elapsed time

# Start/stop stream button
if st.button("Start Stream"):
    st.session_state.streaming = True
    st.session_state.start_time = time.time()  # Set start time
    st.session_state.last_elapsed_time = None  # Reset last elapsed time

if st.button("Stop Stream"):
    st.session_state.streaming = False
    if st.session_state.start_time:
        # Save the elapsed time when stopping
        st.session_state.last_elapsed_time = time.time() - st.session_state.start_time
    st.session_state.start_time = None  # Reset start time

# Output folder for the frames
output_folder = "st_to_txt/data/saved_stream_frames"  # Set folder name

# Video capture object
cap = cv2.VideoCapture(0)  # 0 for the webcam

# Check if the video has been opened
if not cap.isOpened():
    st.error("Error accessing the webcam.")

# Query the FPS of the stream
source_fps = cap.get(cv2.CAP_PROP_FPS)
st.write(f"FPS of the stream: {source_fps}")

# Desired save FPS
save_fps = 25

# Create a timer placeholder that updates
timer_placeholder = st.empty()

# Display video if streaming is active
if st.session_state.streaming:
    stframe = st.empty()  # Placeholder for the video

    while st.session_state.streaming:
        # Calculate elapsed time
        elapsed_time = time.time() - st.session_state.start_time  # Time in seconds

        # Update the timer line instead of creating a new one
        timer_placeholder.write(f"Stream running for: {elapsed_time:.2f} seconds")

        # Read frame
        ret, frame = cap.read()
        if not ret:
            st.error("Error retrieving the video stream.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save frame, only 25 frames per second
        st.session_state.frame_count = save_frames_from_stream(
            frame, 
            output_folder=output_folder, 
            source_fps=int(source_fps),  # FPS of the stream
            save_fps=save_fps,  # Target FPS
            frame_size=(210, 260), 
            frame_count=st.session_state.frame_count
        )

        # Display video in the Streamlit interface
        stframe.image(frame_rgb, caption="Live Video", use_container_width=True)

    cap.release()
else:
    # Show the last elapsed time if the stream was stopped
    if st.session_state.last_elapsed_time is not None:
        timer_placeholder.write(
            f"The stream last ran for: {st.session_state.last_elapsed_time:.2f} seconds"
        )
    else:
        timer_placeholder.write("The stream is stopped. No time recorded.")


# Translation section
st.title("Translation 📕")
st.write("The weather will be stormy today.")