import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 130)
engine.setProperty("volume", 1.0)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

# Open camera
cap = cv2.VideoCapture(0)

# Cooldown timer for speech
last_speak_time = time.time()
last_spoken = None

# Store last 5 gestures to stabilize detection
gesture_history = []

def recognize_gesture(landmarks):
    """Recognizes hand gestures based on finger positions"""
    finger_states = []  # 1 if extended, 0 if folded

    # Thumb detection (checks both x and y position)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    if thumb_tip.y < thumb_mcp.y and abs(thumb_tip.x - thumb_mcp.x) > 0.05:
        finger_states.append(1)  # Thumb extended
    else:
        finger_states.append(0)  # Thumb folded

    # Other fingers (checks tip vs PIP joint, with threshold)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip].y < (landmarks[pip].y - 0.03):  # More lenient threshold
            finger_states.append(1)  # Finger extended
        else:
            finger_states.append(0)  # Finger folded

    # Convert to string pattern
    gesture_pattern = "".join(map(str, finger_states))

    # Additional check for "No" (Index + Middle Extended, Close Together)
    if finger_states == [0, 1, 1, 0, 0]:  
        index_x = landmarks[8].x
        middle_x = landmarks[12].x
        if abs(index_x - middle_x) < 0.03:  # Fingers should be very close
            return "No"

    # Sign language gestures
    gestures = {
        "11111": "Hello",        # Open Palm
        "00000": "Yes",          # Fist
        "11000": "Thumbs Up",    # Only Thumb Extended
        "00100": "Peace",        # Index + Middle Extended (V Sign)
        "01001": "I Love You",   # Index + Pinky + Thumb Up
    }

    return gestures.get(gesture_pattern, "Unknown Gesture")

# Start Video Processing
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Convert back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks.landmark)

                # Stabilization: Store last 5 gestures
                gesture_history.append(gesture)
                if len(gesture_history) > 5:
                    gesture_history.pop(0)

                # Speak only if the last 5 frames detected the same gesture
                if len(set(gesture_history)) == 1 and gesture != last_spoken and gesture != "Unknown Gesture":
                    if time.time() - last_speak_time > 2:  # 2-second cooldown
                        engine.say(gesture)
                        engine.runAndWait()
                        last_spoken = gesture
                        last_speak_time = time.time()

                # Display the recognized gesture
                cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video feed
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
