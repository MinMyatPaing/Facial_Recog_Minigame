import os
import cv2
import time
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the pre-trained model
model = load_model("best_model.h5.keras")

# Load the Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion tracking variables
emotion_time_counters = {emotion: 0 for emotion in ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')}
start_time = time.time()
capture_duration = 5  # Capture emotions for 5 seconds

previous_frame_time = time.time()  # Initialize the previous_frame_time

# Emoji paths based on emotions
emoji_images = {
    'angry': './emoji images/Angry Emoji [Free Download iPhone Emojis in PNG].png',
    'disgust': './emoji images/Poisoned Emoji [Free Download iPhone Emojis].png',
    'fear': './emoji images/Omg Emoji [Free Download iPhone Emojis].png',
    'happy': './emoji images/Smiling Face Emoji.png',
    'sad': './emoji images/Crying Emoji [Download iPhone Emojis].png',
    'surprise': './emoji images/Surprised Emoji [Free Download IOS Emojis].png',
    'neutral': './emoji images/Neutral Emoji [Free Download iPhone Emojis].png',
}

def display_emoji(emotion):
    """Display the corresponding emoji image for the given emotion."""
    image_path = emoji_images.get(emotion, './emoji images/Neutral Emoji [Free Download iPhone Emojis].png')
    emoji_img = cv2.imread(image_path)
    resized_img = cv2.resize(emoji_img, (500, 500))  # Resize emoji image for display
    cv2.imshow(emotion.capitalize(), resized_img)  # Display the emoji image

def start_scanner():
    """Start the emotion detection scanner."""
    global previous_frame_time  # Declare the global variable
    cap = cv2.VideoCapture(0)

    global start_time, emotion_time_counters
    start_time = time.time()  # Reset the timer

    while True:
        ret, test_img = cap.read()  # Capture frame
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  # Detect faces

        current_frame_time = time.time()
        frame_time = current_frame_time - previous_frame_time  # Time taken to process one frame in seconds
        frame_time_ms = frame_time * 1000  # Convert frame time to milliseconds
        previous_frame_time = current_frame_time  # Update previous_frame_time

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)  # Draw rectangle around the face
            roi_gray = gray_img[y:y + w, x:x + h]  # Crop the region of interest (face area)
            roi_gray = cv2.resize(roi_gray, (224, 224))  # Resize the face to 224x224 pixels
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # Normalize the image

            predictions = model.predict(img_pixels)  # Predict the emotion

            # Find the max index of the predictions
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]  # Get the predicted emotion

            # Update the corresponding emotion counter based on frame processing time
            emotion_time_counters[predicted_emotion] += frame_time_ms  # Add frame time in milliseconds to emotion

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))  # Resize the image for display
        cv2.imshow('Facial emotion analysis', resized_img)  # Display the image

        key = cv2.waitKey(10)

        # Check if the time is up (5 seconds have passed)
        if time.time() - start_time >= capture_duration:
            # Determine the emotion with the highest time accumulated
            most_frequent_emotion = max(emotion_time_counters, key=emotion_time_counters.get)
            print(f"The most frequent emotion detected over 5 seconds is: {most_frequent_emotion}")

            # Show the most frequent emotion as an emoji
            display_emoji(most_frequent_emotion)

            # Wait for the user to press a key to go back to scanning
            print("Press 'r' to return to scanning or 'q' to quit.")
            while True:
                key = cv2.waitKey(0)
                if key == ord('r'):  # Return to scanning
                    emotion_time_counters = {emotion: 0 for emotion in emotions}  # Reset counters
                    start_time = time.time()  # Reset the start time
                    previous_frame_time = start_time  # Initialize previous_frame_time for the new session
                    break
                elif key == ord('q'):  # Quit the program
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Start the scanner
start_scanner()