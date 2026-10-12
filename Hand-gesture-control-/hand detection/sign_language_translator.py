import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "sign_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\n"
        "Check the location of sign_model.joblib."
    )

model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")

# ==========================================
# 2. SET UP MEDIAPIPE HAND DETECTION
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# 3. OPEN WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    hands.close()
    raise RuntimeError(
        "Cannot open webcam. Close other camera apps "
        "or try changing VideoCapture(0) to VideoCapture(1)."
    )

# ==========================================
# 4. INITIALIZE TEXT VARIABLES
# ==========================================

predicted_sign = "Show a hand sign"
confidence_text = ""
recognized_text = ""

valid_signs = {"HI", "OK", "HOW ARE YOU", "THANK YOU"}

print("\nAI Sign Language Translator started.")
print("S = Save the predicted letter")
print("C = Clear recognized text")
print("Q = Quit")

# ==========================================
# 5. MAIN WEBCAM LOOP
# ==========================================

try:
    while True:

        success, frame = camera.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb_frame)

        predicted_sign = "No hand detected"
        confidence_text = ""

        # ======================================
        # 6. EXTRACT LANDMARKS AND PREDICT SIGN
        # ======================================

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks, each with x, y, z
            features = []

            for landmark in hand_landmarks.landmark:
                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Convert to 63 model features
            input_data = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)

            # Predict letter
            prediction = model.predict(input_data)[0]
            predicted_sign = str(prediction)

            # Calculate confidence when supported
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = float(np.max(probabilities)) * 100

                confidence_text = (
                    f"Confidence: {confidence:.1f}%"
                )

        # ======================================
        # 7. DISPLAY PREDICTION AND TEXT
        # ======================================

        # Dark background for text
        cv2.rectangle(
            frame,
            (10, 10),
            (620, 175),
            (30, 30, 30),
            -1
        )

        # Current predicted sign
        cv2.putText(
            frame,
            f"Sign: {predicted_sign}",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Confidence
        if confidence_text:
            cv2.putText(
                frame,
                confidence_text,
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Accumulated letters
        cv2.putText(
            frame,
            f"Text: {recognized_text[-25:]}",
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Instructions
        cv2.putText(
            frame,
            "S: Save letter | C: Clear | Q: Quit",
            (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        # Show webcam window
        cv2.imshow(
            "AI Sign Language Translator",
            frame
        )

        # ======================================
        # 8. HANDLE KEYBOARD INPUT
        # ======================================

        key = cv2.waitKey(1) & 0xFF

        # Press S to save the current prediction
        if key == ord("s"):
            if predicted_sign in valid_signs:
                recognized_text += predicted_sign
                print(
                    f"Added: {predicted_sign} | "
                    f"Text: {recognized_text}"
                )
            else:
                print("No valid sign to save. Show a trained sign.")

        # Press C to clear accumulated text
        elif key == ord("c"):
            recognized_text = ""
            print("Recognized text cleared.")

        # Press Q to quit
        elif key == ord("q"):
            break

# ==========================================
# 9. RELEASE RESOURCES
# ==========================================

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()

    print("\nFinal recognized letters:", recognized_text)
    print("Sign Language Translator closed.")