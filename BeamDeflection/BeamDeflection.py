import cv2
import numpy as np
import pickle
from fenics import *
import matplotlib.pyplot as plt
import tkinter as tk

# Define displacement boundary condition
def displacement_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

# Define strain and stress
def epsilon(u):
    return sym(grad(u))

def sigma(G, lmbda, u):
    return lmbda*div(u)*Identity(len(u)) + 2*G*epsilon(u)

def calculate_fem(deflection_mm):
       # Define the geometry and mesh
        L = 2.4  # Length of the beam
        H = 0.03  # Height of the beam
        W = 0.005 # Width of the beam
        num_elements = 10  # Number of elements

        # Create mesh and define function space
        mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), num_elements, num_elements, num_elements)
        V = VectorFunctionSpace(mesh, 'P', 1)

        bc = DirichletBC(V, Constant((0, 0, 0)), displacement_boundary)

        # Define material properties
        E = 2400e9  # Young's modulus
        nu = 0.37   # Poisson's ratio
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = Constant(mu)
        lmbda = Constant(lmbda)

        # Define problem
        u = TrialFunction(V)
        v = TestFunction(V)
        displacement = Constant((0, 0, deflection_mm))  # Displacement applied at one end

        a = inner(sigma(G, lmbda, u), epsilon(v))*dx
        L = dot(displacement, v)*ds  # Use ds for surface integration

        # Compute solution
        u = Function(V)
        solve(a == L, u, bc)

        # Compute von Mises stress
        V = FunctionSpace(mesh, 'P', 1)
        von_Mises = project(sqrt(3.0 / 2.0 * inner(sigma(G, lmbda, u) - (1. / 3) * tr(sigma(G, lmbda, u)) * Identity(len(u)), sigma(G, lmbda, u) - (1. / 3) * tr(sigma(G, lmbda, u)) * Identity(len(u)))), V)

        # Display von Mises stress with contour plot
        plt.figure()
        contour = plot(von_Mises, title='von Mises Stress', cmap='jet')  # Adjust the colormap to resemble MATLAB's default
        plt.colorbar(contour, label='von Mises Stress')
        plt.show()

# Get radius from user
def get_circle_radius():
    # Create the Tkinter root window
    root = tk.Tk()
    # Hide the root window
    root.withdraw() 
    # Create simpledialog window for radius
    radius = tk.simpledialog.askinteger("Enter Radius", "Enter the radius of the circle [mm]:")
    # Close the Tkinter root window
    root.destroy()  
    # Handle the input
    if radius is not None:
        print(f"Radius entered: {radius} mm")
        return radius
    else:
        print("Invalid input or canceled.")
        return None

# Function to reset color selection
def reset_color_selection():
    global hsv_lower_bound, hsv_upper_bound, selecting_color, initial_position
    hsv_lower_bound = np.array([0, 0, 0])
    hsv_upper_bound = np.array([179, 255, 255])
    selecting_color = True
    initial_position = None
    
# Function to select the color range
def select_color(event, x, y, flags, param):
    global frame, hsv_lower_bound, hsv_upper_bound, selecting_color

    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting_color:
            # Extract the region of interest (ROI)
            roi = frame[y-5:y+5, x-5:x+5]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate the minimum and maximum HSV values in the ROI
            min_hsv = np.min(hsv_roi, axis=(0, 1))
            max_hsv = np.max(hsv_roi, axis=(0, 1))
            
            # Set lower and upper bounds, broader than original range
            hsv_lower_bound = np.array([max(0, min_hsv[0] - 10), max(0, min_hsv[1] - 40), max(0, min_hsv[2] - 40)])
            hsv_upper_bound = np.array([min(179, max_hsv[0] + 10), min(255, max_hsv[1] + 40), min(255, max_hsv[2] + 40)])

            # Set flag to stop color selection
            selecting_color = False
        else:
            reset_color_selection()

# Load the calibration parameters
with open("CameraCalibration/calibration.pkl", "rb") as f:
    cameraMatrix, dist = pickle.load(f)

# Capture video using the camera
cap = cv2.VideoCapture(0)
initial_position = None  
selecting_color = True

# Initialize lower and upper bounds
hsv_lower_bound = np.array([0, 0, 0])
hsv_upper_bound = np.array([179, 255, 255])

# Set mouse callback
cv2.namedWindow('Marker Detection')
cv2.setMouseCallback('Marker Detection', select_color)

while True:
    # Read frames
    ret, oldframe = cap.read()
    
    # Check if the frame is captured successfully
    if not ret:
        print("Error capturing frame")
        break
    
    h,  w = oldframe.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    # Undistort the frame using cv2.undistort
    dst = cv2.undistort(oldframe, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    frame = dst[y:y+h, x:x+w]

    if selecting_color:
        cv2.putText(frame, "Select color by clicking:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # HSV image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image based on the selected color range
        mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find and show, largest contour found
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the center and radius of the bounding circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center, radius = (int(x), int(y)), int(radius)

            # Draw enclosing circle and initial circle on the frame
            cv2.circle(frame, center, radius, (255, 0, 0), 2)
            cv2.circle(frame, initial_position, 5, (0, 255, 0), -1) 

            # Set initial position only on first loop
            if initial_position is None:
                # Get radius from user:
                known_radius_mm = get_circle_radius()
                if known_radius_mm is None: 
                    known_radius_mm = 20 # default radius of 20mm
                initial_position = center
                known_radius_pixels = radius
            else:
                # Get current position, mark it
                current_position = center
                cv2.circle(frame, current_position, 5, (255, 255, 0), -1) 

                # Get deflection value for x and y in pixels
                deflection_x_pixels = current_position[0] - initial_position[0]
                deflection_y_pixels = current_position[1] - initial_position[1]

                # Calculate mm per pixel
                mm_per_pixel = known_radius_mm / known_radius_pixels

                # Remap pixel -> mm for x and y
                deflection_x_mm = deflection_x_pixels * mm_per_pixel
                deflection_y_mm = deflection_y_pixels * mm_per_pixel

                # Calculate total deflection
                deflection_mm = np.sqrt(deflection_x_mm ** 2 + deflection_y_mm ** 2)

                # Draw line between current and initial position
                cv2.line(frame, initial_position, current_position, (255, 255, 0), 2) 

                # Find midpoint of the line
                midpoint = ((initial_position[0] + current_position[0]) // 2, (initial_position[1] + current_position[1]) // 2)

                # Put deflection text on the midpoint
                deflection_text = f"Deflection: {deflection_mm:.2f} mm"
                cv2.putText(frame, deflection_text, (midpoint[0] + 10, midpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    
    # Show result
    cv2.imshow('Marker Detection', frame)

    key = cv2.waitKey(1) & 0xFF  
    if key == ord('q'): # Quit on Q button press
        break
    if key == ord('f'): # Fem on F button press
        if not selecting_color:
            calculate_fem(deflection_mm)
        else:
            print("Object was not picked!")
    if key == ord('c'): # Check camera calibration o C press
        cv2.imshow('Original', oldframe) # Orginal frame
        cv2.imshow('Undistorted', frame) # Removed distortion frame
    if key == ord('p'): # Print to terminal on P press
        if not selecting_color:
            print(f"Deflection x: {deflection_x_mm:.2f} mm") # Print x deflection
            print(f"Deflection y: {deflection_y_mm:.2f} mm") # Print y deflection
            print(f"Deflection mag: {deflection_mm:.2f} mm") # Print total deflection
        else:
            print("Object was not picked!")        

# Cleanup routine
cap.release()
cv2.destroyAllWindows()