import streamlit as st
import cv2
import os
from utils.stream_to_frames import save_frames_from_stream
import time  # Import for time measurement

# Title and description
st.title("Talk to the Hand 👋🏽")
st.write("Here you can view the livestream.")

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
output_folder = "data/saved_stream_frames"  # Set folder name

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
