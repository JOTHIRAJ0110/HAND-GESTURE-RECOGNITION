import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gesture recognition function
def recognize_gesture(hand_landmarks):
    """
    Recognize hand gestures based on landmark positions
    """
    # Fingertip landmarks
    fingertips = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

# Get y-coordinates of fingertips
    tips_y = [hand_landmarks.landmark[tip].y for tip in fingertips]

    # Get y-coordinates of the bases of fingers (MCP joints)
    bases_y = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,  # Thumb base
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,  # Index base
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,  # Middle base
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,  # Ring base
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y  # Pinky base
    ]

    # Thumb x and y coordinates
    thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

    # 1. THUMBS UP
    if all(tips_y[i] < tips_y[i + 1] for i in range(len(tips_y) - 1)):
        return "THUMBS UP"

    # 2. CLOSED FIST (all fingertips below their respective bases)
    if all(tips_y[i] > bases_y[i] for i in range(len(tips_y))):
        return "CLOSED FIST"

    # 3. OPEN FIST (all fingertips above their respective bases)
    if all(tips_y[i] < bases_y[i] for i in range(len(tips_y))):
        return "OPEN FIST"

    # 4. PEACE SIGN (index and middle fingers extended, others down)
    threshold = 0.02  # Allow some margin for gesture detection
    if (tips_y[0] > tips_y[1] + threshold and  # Thumb down
            tips_y[1] < bases_y[1] - threshold and  # Index finger extended
            tips_y[2] < bases_y[2] - threshold and  # Middle finger extended
            tips_y[3] > bases_y[3] + threshold and  # Ring finger down
            tips_y[4] > bases_y[4] + threshold):  # Pinky finger down
        return "PEACE"

    # Default gesture if no match
    return "UNKNOWN"


# Main hand tracking and gesture recognition
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                print("Error: Unable to read from camera")
                break

            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame and find hands
            results = hands.process(frame_rgb)

            # Draw hand annotations and recognize gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    # Recognize and display gesture
                    gesture = recognize_gesture(hand_landmarks)
                    cv2.putText(frame, gesture, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hands detected", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
	
