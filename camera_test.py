import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    print("Check camera permissions or try changing 0 to 1.")
    exit()

print("Webcam started successfully!")
print("Press Q to close the camera.")

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Cannot read camera frame.")
        break

    # Mirror the camera view
    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "AI Sign Language Translator",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Webcam Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()