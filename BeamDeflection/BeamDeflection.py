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
    return lmbda*div(u)*Identity(2) + 2*G*epsilon(u)

def calculate_fem(deflection_mm):
    # Define the geometry and mesh (in millimeters)
    L = 240.0  # Length of the beam
    H = 30.0    # Height of the beam
    num_elements = [30,10]  # Number of elements

    # Create mesh and define function space (2D)
    mesh = RectangleMesh(Point(0, 0), Point(L, H), num_elements[0], num_elements[1])
    V = VectorFunctionSpace(mesh, 'P', 1)

    bc = DirichletBC(V, Constant((0, 0)), displacement_boundary)

    # Define material properties
    E = 3500e6  # Young's modulus (in MPa)
    nu = 0.35   # Poisson's ratio
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = Constant(mu)
    lmbda = Constant(lmbda)

    # Define problem
    u = TrialFunction(V)
    v = TestFunction(V)
    displacement = Constant((0, deflection_mm))  # Displacement applied at one end 
    a = inner(sigma(G, lmbda, u), epsilon(v))*dx
    L = dot(displacement, v)*ds  # Use ds for surface integration

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Compute von Mises stress
    V = FunctionSpace(mesh, 'P', 1)
    von_Mises = project(sqrt(3.0 / 2.0 * inner(sigma(G, lmbda, u) - (1. / 3) * tr(sigma(G, lmbda, u)) * Identity(2), sigma(G, lmbda, u) - (1. / 3) * tr(sigma(G, lmbda, u)) * Identity(2))), V)

    # Display von Mises stress with contour plot
    plt.figure()
    mngr = plt.get_current_fig_manager()
    mngr.canvas.manager.window.wm_geometry("+%d+%d" % (200, 200))
    contour = plot(von_Mises, title='von Mises Stress', cmap='jet')  # Adjust the colormap to resemble MATLAB's default
    plt.colorbar(contour, label='von Mises Stress')
    plt.show()

# Open new tk root window
def open_tk_window():
    # Create the Tkinter root window
    root = tk.Tk()
    # Update the display and handle events
    root.update_idletasks()
    # Center root window
    root.tk.eval(f'tk::PlaceWindow {root._w} center')
    # Return root window
    return root
    
    
# Get radius from user
def get_circle_radius():
    # Open window
    root = open_tk_window()
    # Hide the root window
    root.withdraw()
    # Create simpledialog window for radius
    radius = tk.simpledialog.askinteger("Enter Radius", "Enter the radius of the circle [mm]:", parent=root) 
    # Destroy window
    root.destroy()
    # Handle the input
    if radius is not None:
        print(f"Radius entered: {radius} mm")
        return radius
    else:
        print("Invalid input or canceled.")
        return None

# Function to set line/text color 
def choose_drawing_colors():
    # BGR colors
    preset_colors = {
        'Blue': (255, 0, 0),
        'Green': (0, 255, 0),
        'Red': (0, 0, 255),
        'Yellow': (0, 255, 255),
        'Purple': (128, 0, 128),
        'Orange': (0, 165, 255),
    }

    # Get keys from color options dict
    preset_vars = list(color_vars.keys())

    # Open root window
    root = open_tk_window()
    root.title("Choose Color and Variable")

    # Set default color and var
    selected_color, selected_var = tk.StringVar(), tk.StringVar()
    selected_color.set("Blue")
    selected_var.set("Enclosing Circle")

    # Option menu for colors
    color_menu = tk.OptionMenu(root, selected_color, *preset_colors.keys())
    color_menu.pack()

    # Option menu for vars
    var_menu = tk.OptionMenu(root, selected_var, *preset_vars)
    var_menu.pack()

    # Submit button
    ok_button = tk.Button(root, text="OK", command=root.destroy)
    ok_button.pack()

    # Make the root window modal
    root.grab_set()

    # Continue updating while the window exists
    root.mainloop()

    return preset_colors.get(selected_color.get()), selected_var.get()

# Function to set line/text width 
def choose_drawing_width():
    root = open_tk_window()
    # Hide the root window
    root.withdraw()
    width = tk.simpledialog.askinteger("Drawing Width", "Enter the drawing width:", parent=root)
    # Destroy window
    root.destroy()
    # Handle the input
    if width is not None:
        print(f"Width entered: {width} mm")
        return width if 0 < width < 10 else 2
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

# Function to display help text on the OpenCV frame
def display_multiline_text(frame, text, position, screenx, font_scale=0.5, font_thickness=1, font_color=(255, 255, 255)):
    # Font and position set
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = position
    # Enumerate through text and put lines dy under each other
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (screenx, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Display help
def display_help_on_frame(frame, show=False):
    if show:
        help_text = """
        Press 'q' to quit.
        Press 'f' to calculate fem.
        Press 'p' to print deflection values.
        Press 'r' to get radius value.
        Press 'c' to set color values.
        Press 'w' to set width value.
        """
    else:
        help_text = "Press 'h' to display help."

    display_multiline_text(frame, help_text, (20, 20), 10)

# Display deflection text
def display_deflection_components_on_frame(frame, deflection_x_mm, deflection_y_mm):
    deflection_text = f"""
        Deflection x: {deflection_x_mm:.2f} mm
        Deflection y: {deflection_y_mm:.2f} mm
    """
    display_multiline_text(frame, deflection_text, (20, 20), 360)

# Load the calibration parameters
with open("CameraCalibration/calibration.pkl", "rb") as f:
    cameraMatrix, dist = pickle.load(f)

# Capture video using the camera
cap = cv2.VideoCapture(0)
initial_position = None  
selecting_color = True
show_help, show_deflection = False, False
known_radius_mm = 20 # default radius of 20mm 

# Default colors for lines, text and their width:
color_vars = {  'Enclosing Circle' : (255, 0, 0),
                'Initial Point' : (0, 255, 0),
                'Current Point' : (255, 255, 0),
                'Deflection Line' : (255, 0, 255), 
                'Deflection Text' : (255, 255, 0)}
width = 2

# Initialize lower and upper bounds
hsv_lower_bound = np.array([0, 0, 0])
hsv_upper_bound = np.array([179, 255, 255])

# Set mouse callback
cv2.namedWindow('Marker Detection')
# Adjust the initial window position
cv2.moveWindow('Marker Detection', 200, 140)  
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

            # Draw enclosing circle
            cv2.circle(frame, center, radius, color_vars['Enclosing Circle'], width)

            # Check if you should show help
            display_help_on_frame(frame, show_help)

            # Show deflection components when set 
            if show_deflection:
                display_deflection_components_on_frame(frame, deflection_x_mm, deflection_y_mm)

            # Set initial position only on first loop
            if initial_position is None:
                initial_position = center
                known_radius_pixels = radius
            else:
                # Get current position
                current_position = center

                # Draw line between current and initial position
                cv2.line(frame, initial_position, current_position, color_vars['Deflection Line'], width) 

                # Find midpoint of the line
                midpoint = ((initial_position[0] + current_position[0]) // 2, (initial_position[1] + current_position[1]) // 2)

                # Draw initial and current position
                cv2.circle(frame, initial_position, width, color_vars['Initial Point'], -1) 
                cv2.circle(frame, current_position, width, color_vars['Current Point'], -1)            

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

                # Put deflection text on the midpoint
                deflection_text = f"Deflection: {deflection_mm:.2f} mm"
                cv2.putText(frame, deflection_text, (midpoint[0] + 10, midpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_vars['Deflection Text'], width)
    
    # Show result
    cv2.imshow('Marker Detection', frame)

    key = cv2.waitKey(1) & 0xFF  
    if key == ord('q'): # Quit on Q button press
        break
    elif key == ord('f'): # Fem on F button press
        if not selecting_color:
            calculate_fem(deflection_y_mm)
        else:
            print("Object was not picked!")
    elif key == ord('p'): # Print deflection components on P press
        if not selecting_color:
            show_deflection = not show_deflection
        else:
            print("Object was not picked!")      
    elif key == ord('r'): # Get radius value from user on R press
         known_radius_mm = get_circle_radius()
         if not known_radius_mm or known_radius_mm <= 0:
           known_radius_mm = 20
    elif key == ord('c'): # Set colors on C press
        color, var = choose_drawing_colors()
        if color and var:
            color_vars[var] = color
    elif key == ord('w'): # Set width on W press
        width = choose_drawing_width()
    elif key == ord('h'): # Show help on H press
        show_help = not show_help  # Toggle the flag

# Cleanup routine 
cap.release()
cv2.destroyAllWindows()