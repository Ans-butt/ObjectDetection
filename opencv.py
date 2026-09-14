from ultralytics import YOLO
import cv2


# Load YOLO26 model
model = YOLO("yolo26s.pt")

# Open laptop camera
cap = cv2.VideoCapture(0)

# Check camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started successfully!")
print("Press 'q' to quit.")


while True:

    # Read frame from camera
    success, frame = cap.read()

    if not success:
        print("Error: Could not read camera frame.")
        break

    # Object detection + tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        tracker="bytetrack.yaml"
    )

    # Draw boxes, labels and tracking IDs
    annotated_frame = results[0].plot()

    # Display camera
    cv2.imshow(
        "YOLO26 Object Detection & Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
cap.release()
cv2.destroyAllWindows()