import cv2
import numpy as np

# Capture video using the camera
cap = cv2.VideoCapture(0)
# Initialize initial position
initial_position = None  
# Initialize the width of the marker
known_width_mm = 15

while True:
    ret, frame = cap.read()

    # Convert frame to HSV (Hue, Saturation, Value) for better color handling
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for yellow color
    lower_bound = np.array([20, 80, 80])	 
    upper_bound = np.array([30, 255, 255])

    # Threshold the HSV image to get only the yellow color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if contours:
        # Get the largest contour (assuming it's the end of the beam)
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Get the center of the detected area
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            # coordinates of centroid
            cx = int(M["m10"] / M["m00"])  
            cy = int(M["m01"] / M["m00"]) 

            # Display the centroid on the frame
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            if initial_position is None:
                # Set initial position
                initial_position = (cx, cy)
                # Set width of the bounding rectangle in pixels
                known_width_pixels = w 
            else:
                current_position = (cx, cy)
                # Draw a red circle at the initial position
                cv2.circle(frame, initial_position, 5, (0, 0, 255), -1) 
                # Draw a line between initial and current positions
                cv2.line(frame, initial_position, current_position, (0, 0, 255), 2)

                # Calculate the change in position (deflection) in pixels
                deflection_pixels = np.sqrt((current_position[0] - initial_position[0]) ** 2 +
                                            (current_position[1] - initial_position[1]) ** 2)

                # Compute pixels per millimeter conversion factor
                pixels_per_mm = known_width_pixels / known_width_mm

                # Convert deflection from pixels to millimeters
                deflection_mm = deflection_pixels / pixels_per_mm
                # Calculate the midpoint of the line
                midpoint = ((initial_position[0] + current_position[0]) // 2, (initial_position[1] + current_position[1]) // 2)

                # Display deflection value next to the line
                deflection_text = f"Deflection: {deflection_mm:.2f} mm"
                cv2.putText(frame, deflection_text, (midpoint[0] + 10, midpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display the frame with detected end of the beam
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()