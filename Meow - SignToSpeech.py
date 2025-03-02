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
engine.setProperty("voice", voices[0].id)  # Adjust index if needed

# Open camera
cap = cv2.VideoCapture(0)

# Variables to prevent frequent speaking
last_spoken = None
last_speak_time = time.time()

def recognize_sign_language(landmarks):
    """ Recognize common sign language gestures based on finger positions """

    try:
        # Finger states: 1 = Up, 0 = Down
        finger_states = []

        # Thumb detection using Y-coordinates
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]  # MCP joint (closer to palm)

        if thumb_tip.y < thumb_mcp.y:  # Thumb is pointing up
            finger_states.append(1)
        else:
            finger_states.append(0)

        # Other fingers: Tip vs. PIP joint
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip].y < landmarks[pip].y:
                finger_states.append(1)  # Finger is extended
            else:
                finger_states.append(0)

        # Convert pattern to a string
        gesture_pattern = "".join(map(str, finger_states))

        # Sign Language Gestures
        signs = {
            "11111": "Hello 👋",
            "11000": "Yes 👍",
            "10000": "No 👎",
            "10101": "I Love You 🤟",
            "01111": "Thank You 🙏",
            "00011": "Help 🤲",
            "11110": "Stop ✋",
            "01100": "Goodbye 👋"
        }

        return signs.get(gesture_pattern, "Unknown Sign")
        

    except Exception as e:
        print(f"Error in recognition: {e}")
        return "Error"

# Process video stream
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera not detected.")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Convert back to BGR
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Process detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize sign language gesture
                sign = recognize_sign_language(hand_landmarks.landmark)

                # Speak the recognized gesture (Only if stable for 2 seconds)
                current_time = time.time()
                if sign != last_spoken and sign != "Error" and current_time - last_speak_time > 2:
                    print(f"Detected Sign: {sign}")
                    engine.say(sign)
                    engine.runAndWait()
                    last_spoken = sign
                    last_speak_time = current_time  # Update last spoken time

                # Display the recognized sign
                cv2.putText(frame, f"Sign: {sign}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video feed
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
