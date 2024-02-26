import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Unable to open webcam")
    exit()

# Capture frames from the webcam
while True:
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
