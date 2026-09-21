from ultralytics import YOLO
import cv2
import os
from database import save_detection


# Load YOLO26 model
model = YOLO("yolo26s.pt")

# Create folder for detection images
os.makedirs("images", exist_ok=True)

# Keep track of IDs that have already been saved
saved_track_ids = set()

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

    result = results[0]

    # Get detected boxes
    if result.boxes is not None:

        for box in result.boxes:

            # Make sure this detection has a tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            # Object class
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            # Confidence
            confidence = float(box.conf[0])

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Save only the first detection for each Track ID
            if track_id not in saved_track_ids:

                # Create unique image name
                image_name = f"{object_name}_track_{track_id}.jpg"
                image_path = os.path.join("images", image_name)

                # Save current frame
                cv2.imwrite(image_path, frame)

                # Save detection information to MySQL
                save_detection(
                    object_name=object_name,
                    confidence=confidence,
                    track_id=track_id,
                    image_path=image_path,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    camera_name="Laptop Webcam"
                )

                # Mark this Track ID as saved
                saved_track_ids.add(track_id)

                print(
                    f"Saved: {object_name} | "
                    f"Track ID: {track_id} | "
                    f"Confidence: {confidence:.2f}"
                )

    # Draw boxes, labels and tracking IDs
    annotated_frame = result.plot()

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
