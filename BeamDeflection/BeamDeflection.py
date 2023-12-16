import cv2
import numpy as np
from fenics import *
import matplotlib.pyplot as plt

# Capture video using the camera
cap = cv2.VideoCapture(0)
# Initialize initial position
initial_position = None  
# Initialize the width of the marker
known_width_mm = 22

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
                print("Deflection in mm:", deflection_mm)

    # Display the frame with detected end of the beam
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    key = cv2.waitKey(1) & 0xFF  # Ensure it's an 8-bit integer
    if key == ord('q'):
        break
    if key == ord('f'):
        # Define the geometry and mesh
        L = 2.4  # Length of the beam
        H = 0.03  # Height of the beam
        W = 0.005 # Width of the beam
        num_elements = 10  # Number of elements

        # Create mesh and define function space
        mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), num_elements, num_elements, num_elements)
        V = VectorFunctionSpace(mesh, 'P', 1)

        # Define displacement boundary condition
        def displacement_boundary(x, on_boundary):
            return near(x[0], 0) and on_boundary

        bc = DirichletBC(V, Constant((0, 0, 0)), displacement_boundary)

        # Define material properties
        E = 2400e9  # Young's modulus
        nu = 0.37   # Poisson's ratio
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = Constant(mu)
        lmbda = Constant(lmbda)

        # Define strain and stress
        def epsilon(u):
            return sym(grad(u))

        def sigma(u):
            return lmbda*div(u)*Identity(len(u)) + 2*G*epsilon(u)

        # Define problem
        u = TrialFunction(V)
        v = TestFunction(V)
        displacement = Constant((0, 0, deflection_mm))  # Displacement applied at one end

        a = inner(sigma(u), epsilon(v))*dx
        L = dot(displacement, v)*ds  # Use ds for surface integration

        # Compute solution
        u = Function(V)
        solve(a == L, u, bc)

        # Compute von Mises stress
        V = FunctionSpace(mesh, 'P', 1)
        von_Mises = project(sqrt(3.0 / 2.0 * inner(sigma(u) - (1. / 3) * tr(sigma(u)) * Identity(len(u)), sigma(u) - (1. / 3) * tr(sigma(u)) * Identity(len(u)))), V)

        # Display von Mises stress with contour plot
        plt.figure()
        contour = plot(von_Mises, title='von Mises Stress', cmap='jet')  # Adjust the colormap to resemble MATLAB's default
        plt.colorbar(contour, label='von Mises Stress')
        plt.show()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()