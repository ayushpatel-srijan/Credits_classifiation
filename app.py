import streamlit as st
import cv2
import numpy as np
import os
from keras import models
import shutil
import pytesseract

#from streamlit_player import st_player

st.set_page_config(layout="wide")

model = models.load_model('closing_credits_Resnet50.h5')



def milliseconds_to_timestamp(milliseconds):
    """Convert milliseconds to a timestamp (hours:minutes:seconds.milliseconds).

    Args:
        milliseconds (int): The number of milliseconds to convert.

    Returns:
        str: The timestamp in the format "hours:minutes:seconds.milliseconds".
    """
    milliseconds = int(milliseconds)  # Convert to integer
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return timestamp

def predict_one_frame_per_second(video_path, model):
    # Read the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize variables
    frames = []
    timestamps = []  # List to store the timestamps for each frame

    frames_per_second = int(frame_rate)
    current_second = 0
    org_frame=[]
    texts=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process only one frame per second
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frames_per_second == 0:
            # Perform any necessary pre-processing on the frame before predicting
            # For example, resize the frame to match your model input shape
            # preprocessed_frame = preprocess_frame(frame)

            # Predict using your model (assuming it's a function named `predict`)
            org_frame.append(frame)
         
            
            texts.append(is_text_present_in_frame(frame))
            frame = cv2.resize(frame, (224, 224)) / 255.0
            # preprocessed_frame = frame.reshape(-1, 224, 224, 3)
            frames.append(np.array(frame))
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))  # Save the timestamp for the current frame
            current_second += 1

    cap.release()
    predictions = model.predict(np.array(frames))
    predictions = np.round(predictions, decimals=2)
    return org_frame,predictions, timestamps ,texts

def split_video_by_timestamp(video_path, start_time_ms, end_time_ms, output_file):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

    # Split the video based on the given timestamps
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_time_ms:
            current_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if start_time_ms <= current_timestamp_ms <= end_time_ms:
                out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()

def is_text_present_in_frame(frame):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to extract text from the frame
    text = pytesseract.image_to_string(gray_frame)

    # Check if text is detected in the frame
    return 1 if len(text.strip()) > 0 else 0

def correcting_black_frames(predictions ,texts,frames):
    # converting black frames to credits
    new_pred=[]
    new_texts=[]
    for pred,text,frame in zip(predictions,texts,frames):
        if np.mean(frame)<0.1:
            new_pred.append(np.array([0.0], dtype=np.float32))
            new_texts.append(1)
        else:
            new_texts.append(text)
            new_pred.append(pred)
    return new_pred ,new_texts

def split_video(videoname):
    fn = videoname[:-4]
    if os.path.exists(fn):
        shutil.rmtree(fn)
    os.mkdir(fn)
    frames ,predictions, timestamps ,texts = predict_one_frame_per_second(videoname, model)
    start_time, end_time = None, None
    start_credits_found, end_credits_found = False, False

    predictions ,texts = correcting_black_frames(predictions,texts,frames)

    for ind, (timestamp, prediction, frame) in enumerate(zip(timestamps, predictions, frames)):
        if ind <10: 
            #print(timestamp/1000 , predictions[ind],texts[ind] , sum(texts[ind + 1: ind + 4]))

            if (ind + 4 < len(predictions) and all(np.all(p > 0.2) for p in predictions[ind + 1: ind + 4])) and not start_credits_found and sum(texts[ind -4: ind -1]) >2:
                print(timestamp/1000 , predictions[ind],texts[ind] , sum(texts[ind -4 : ind -1 ]))
                print("Start credits found at timestamp:", milliseconds_to_timestamp(round(timestamp)))
                start_time =timestamp -500
                if start_time < 1000:  # Check if start_time is less than 1 second (1 second = 1000 milliseconds)
                    start_time=0
                    print("setting start time to 0")
                else :
                    start_credits_found = True
                    print("Start time :" ,start_time)
                
                start_credits_found = True
            
        else: 

            if (ind - 10 < len(predictions) and all(np.all(p < 0.2) for p in predictions[ind - 10 : ind - 1])) and not start_credits_found and  sum(texts[ind - 10: ind - 1]) >5:
                print(timestamp/1000 , predictions[ind],texts[ind] , sum(texts[ind + 1: ind + 10]))
                print("Start credits found at timestamp:", milliseconds_to_timestamp(round(timestamp)))
                start_time = timestamp-1000
                start_credits_found = True
                print("Start time :" ,start_time)


            elif(ind + 10 < len(predictions) and all(np.all(p < 0.2) for p in predictions[ind + 1: ind + 10])) and start_credits_found and  sum(texts[ind +1: ind + 10]) >5 :
                print(timestamp/1000 , predictions[ind],texts[ind] , sum(texts[ind + 1: ind + 10]))
                print("End credits found at timestamp:", milliseconds_to_timestamp(round(timestamp)))
                end_time = timestamp
                print("end time :" ,end_time)
                end_credits_found = True
                break


    if not start_credits_found:
        print("No start credits found. Skipping...")
        
    if not end_credits_found:
        print("No end credits found. Skipping...")

    if start_time>1000:
        start_credits_file = 'start_credits.mp4'
        print(f"Splitting starting credits from {0} to {start_time}")
        split_video_by_timestamp(videoname, 0, start_time-1000, start_credits_file)
        shutil.move("start_credits.mp4",os.path.join(fn,'start_credits.mp4'))


    end_credits_file = 'end_credits.mp4'
    print(f"Splitting end credits from {start_time} to {end_time}")

    split_video_by_timestamp(videoname, end_time, float('inf'), end_credits_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    content_file = 'content.mp4'
    print(f"Splitting content from {start_time} to {end_time}")
    split_video_by_timestamp(videoname, start_time, end_time, content_file)

    
    shutil.move("content.mp4",os.path.join(fn,'content.mp4'))
    shutil.move("end_credits.mp4",os.path.join(fn,'end_credits.mp4'))
    print("Video saved in : ",fn)
    return start_time , end_time ,len(frames)

def main():
    st.title("Credits Detection App")
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    end_time =None

    sample = st.selectbox("Sample",[None,"videoplayback",'thailand'])
    if sample:
        fn = sample
        st.session_state.video_path = sample+".mp4"
        st.video( sample+".mp4")
        


    elif uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
        # Save the current uploaded file to the session state
        st.session_state.last_uploaded_file = uploaded_file

        st.session_state.video_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open( st.session_state.video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("Uploaded Video")
        st.video(uploaded_file)

        # Process the video and split
        print("rerunning")
        start_time, end_time, duration = split_video( st.session_state.video_path)
        st.session_state.video_split_done = True
        st.session_state.start_time = start_time
        st.session_state.end_time = end_time
        st.session_state.duration = duration
        fn = uploaded_file.name[:-4]



    elif uploaded_file is not None:
        # Handle the case where the uploaded file is the same as before
        start_time, end_time, duration = st.session_state.start_time, st.session_state.end_time, st.session_state.duration
        fn = uploaded_file.name[:-4]

        #start_time , end_time ,duration = split_video(video_path)


    if end_time or sample:
        if os.path.exists(os.path.join(fn, 'start_credits.mp4')):
            col1, col2 ,col3 = st.columns(3)

            fn = os.path.splitext(os.path.basename(st.session_state.video_path))[0]
            col1.subheader("Start Credits")
            video_file_content = open(os.path.join(fn, 'start_credits.mp4'), 'rb')
            col1.video(video_file_content)

            col2.subheader("Content")
            video_file_content = open(os.path.join(fn, 'content.mp4'), 'rb')
            col2.video(video_file_content)

            col3.subheader("End Credits")
            video_file_credits = open(os.path.join(fn, 'end_credits.mp4'), 'rb')
            col3.video(video_file_credits)

        else:
            col1, col2 = st.columns(2)

            # If no Start Credits, display only two columns
            col1.subheader("Content")
            video_file_content = open(os.path.join(fn, 'content.mp4'), 'rb')
            col1.video(video_file_content,format = "video/mp4")

            col2.subheader("End Credits")
            video_file_credits = open(os.path.join(fn, 'end_credits.mp4'), 'rb')
            col2.video(video_file_credits)

            video_file_content.close()
            video_file_credits.close()

        #with st.expander("Addition feature"):
        #    placeholder = st.empty()
        #    placeholder.video(uploaded_file)

        #    # Handle button states
        #    if st.button("Skip Intro"):
        #        st.session_state.skip_intro = True
        #    if st.button("Skip Credits"):
        #        st.session_state.skip_credits = True
        #    
        #    print(start_time//1000 , duration)
        #    # Update video according to button state
        #    if "skip_intro" in st.session_state and st.session_state.skip_intro:
        #        #placeholder.empty()
        #        placeholder.video(uploaded_file, start_time=int(start_time/1000))
        #    if "skip_credits" in st.session_state and st.session_state.skip_credits:
        #        #placeholder.empty()
        #        placeholder.video(uploaded_file, start_time=int(duration-3))'''

if __name__ == "__main__":
    main()

