import cv2
import mediapipe as mp
import csv
import os

# Project settings
DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, "sign_data.csv")
SAMPLES_PER_SIGN = 200

os.makedirs(DATASET_DIR, exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# CSV columns: label + 21 landmarks * (x, y, z)
columns = ["label"]
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])

# Create CSV with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    hands.close()
    exit()

labels = {
    ord("1"): "HI",
    ord("2"): "OK",
    ord("3"): "HOW ARE YOU",
    ord("4"): "THANK YOU",
}

current_label = None
sample_count = 0
frame_count = 0

print("\nAI Sign Language - Data Collection")
print("Press 1 for sign HI")
print("Press 2 for sign HELLO")
print("Press 3 for sign HOW ARE YOU")
print("Press 4 for sign THANK YOU")


print("Press X to stop collecting a sign")

print("Press Q to quit\n")

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot read webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        if current_label is not None:
            frame_count += 1

            # Save one sample every 5 frames
            if frame_count % 5 == 0:
                coords = []

                for point in hand_landmarks.landmark:
                    coords.extend([point.x, point.y, point.z])

                with open(CSV_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([current_label] + coords)

                sample_count += 1

                if sample_count >= SAMPLES_PER_SIGN:
                    print(
                        f"Finished collecting sign {current_label}"
                    )
                    current_label = None
                    sample_count = 0
                    frame_count = 0

    cv2.putText(
        frame,
        "Sign: " + (current_label or "Not recording"),
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{SAMPLES_PER_SIGN}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Sign Language Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in labels and current_label is None:
        current_label = labels[key]
        sample_count = 0
        frame_count = 0
        print(f"Recording sign {current_label}...")

    elif key == ord("x"):
        current_label = None
        sample_count = 0
        frame_count = 0
        print("Recording stopped.")

    elif key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Data collection closed.")
print("Dataset saved to:", CSV_FILE)