import cv2
import numpy as np
import math

class ImageLineDrawer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        ###########
        # Convert MATLAB intrinsic matrix to OpenCV format
        # MATLAB format: [fx 0 0; 0 fy 0; cx cy 1]
        # OpenCV format: [fx 0 cx; 0 fy cy; 0 0 1]
        
        # Original MATLAB intrinsic matrix
        matlab_intrinsic = np.array([
            [9406.3572, 0, 0],
            [0, 9262.9601, 0], 
            [3163.8267, 1750.1192, 1]
        ])
        
        # Convert to OpenCV format
        fx = matlab_intrinsic[0, 0]
        fy = matlab_intrinsic[1, 1] 
        cx = matlab_intrinsic[2, 0]
        cy = matlab_intrinsic[2, 1]
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Radial distortion coefficients
        # MATLAB: [k1, k2]
        # OpenCV: [k1, k2, p1, p2, k3] where p1,p2 are tangential, k3 is 3rd radial
        radial_distortion = [0.4498, -1.9261]
        distortion_coeffs = np.array([
            radial_distortion[0],  # k1
            radial_distortion[1],  # k2
            0,                     # p1 (tangential)
            0,                     # p2 (tangential) 
            0                      # k3 (3rd order radial)
        ], dtype=np.float32)
        
        # Undistort the image
        self.image = cv2.undistort(self.image, camera_matrix, distortion_coeffs)

        #############
        if self.image is None:
            raise ValueError("Could not load image. Check the file path.")
        
        self.original_image = self.image.copy()
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.lines = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            print(f"Start point: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Show temporary line while dragging
                temp_image = self.image.copy()
                cv2.line(temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Image Line Drawer', temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                
                # Draw the final line
                cv2.line(self.image, self.start_point, self.end_point, (0, 255, 0), 2)
                
                # Calculate length
                length = math.sqrt((self.end_point[0] - self.start_point[0])**2 + 
                                 (self.end_point[1] - self.start_point[1])**2)
                
                # Store line information
                line_info = {
                    'start': self.start_point,
                    'end': self.end_point,
                    'length': length
                }
                self.lines.append(line_info)
                
                print(f"End point: ({x}, {y})")
                print(f"Line length: {length:.2f} pixels")
                print(f"Line #{len(self.lines)} drawn")
                print("-" * 30)
                
                cv2.imshow('Image Line Drawer', self.image)
    
    def run(self):
        cv2.imshow('Image Line Drawer', self.image)
        cv2.setMouseCallback('Image Line Drawer', self.mouse_callback)
        
        print("Instructions:")
        print("- Click and drag to draw lines")
        print("- Press 'c' to clear all lines")
        print("- Press 'r' to reset to original image")
        print("- Press 's' to save current image")
        print("- Press 'l' to list all lines")
        print("- Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear all lines
                self.image = self.original_image.copy()
                self.lines = []
                cv2.imshow('Image Line Drawer', self.image)
                print("All lines cleared")
            elif key == ord('r'):
                # Reset to original
                self.image = self.original_image.copy()
                self.lines = []
                cv2.imshow('Image Line Drawer', self.image)
                print("Reset to original image")
            elif key == ord('s'):
                # Save current image
                cv2.imwrite('image_with_lines.png', self.image)
                print("Image saved as 'image_with_lines.png'")
            elif key == ord('l'):
                # List all lines
                print("\nAll drawn lines:")
                for i, line in enumerate(self.lines, 1):
                    print(f"Line {i}: {line['start']} -> {line['end']}, Length: {line['length']:.2f} pixels")
                print("-" * 30)
        
        cv2.destroyAllWindows()
        return self.lines

# Usage
if __name__ == "__main__":
    try:
        # Replace 'your_image.jpg' with your image path
        drawer = ImageLineDrawer('/Users/cmbreen/Documents/snow/alaska_dataset/all_images/snowfree_photos/CP_final_448/CP DB 6/CP DB 6_WSCT1110.JPG')
        lines = drawer.run()
        
        # Print final summary
        print(f"\nFinal summary: {len(lines)} lines drawn")
        for i, line in enumerate(lines, 1):
            print(f"Line {i}: Start{line['start']} -> End{line['end']}, Length: {line['length']:.2f}px")
            
    except ValueError as e:
        print(f"Error: {e}")