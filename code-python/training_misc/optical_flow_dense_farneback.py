import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt 


def start_dense_optical_flow(video_path, save_video=False, output_path="optical_flow_dense_output.mp4", roi=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Set default ROI if not provided
    if roi is None:
        roi = (0, 0, width, height)
    
    x, y, w, h = roi
    print(f"ROI: x={x}, y={y}, width={w}, height={h}")
    
    # Target FPS for output
    target_fps = 30
    
    # Set up video writer if saving
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Scale dimensions for output
        out_width = int(width * 0.4)
        out_height = int(height * 0.4)
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (out_width, out_height))
        
        if not out.isOpened():
            print("Error: Could not open video writer")
            cap.release()
            return
    
    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    saved_count = 0
    
    # Calculate which frames to save based on fps ratio
    frame_interval = fps / target_fps
    next_frame_to_save = 0
    
    # Arrow step size (sample every N pixels)
    step = 20
    # Motion threshold to reduce clutter
    motion_threshold = 0.5
    
    print(f"Processing video at {fps} fps, saving at {target_fps} fps (interval: {frame_interval:.2f})...")
    print("Press 'Esc' to stop early")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract ROI region for optical flow calculation
        roi_old_gray = old_gray[y:y+h, x:x+w]
        roi_frame_gray = frame_gray[y:y+h, x:x+w]
        
        # Calculate dense optical flow using Farneback method (only on ROI)
        flow = cv2.calcOpticalFlowFarneback(
            roi_old_gray, roi_frame_gray,
            None,
            pyr_scale=0.5,      # Pyramid scale
            levels=3,            # Number of pyramid layers
            winsize=10,          # Averaging window size
            iterations=3,        # Number of iterations at each pyramid level
            poly_n=5,            # Size of pixel neighborhood
            poly_sigma=1.2,      # Standard deviation for Gaussian
            flags=0
        )
        
        # Create visualization on full frame
        flow_viz = frame.copy()
        
        # Draw arrows for the flow vectors (only within ROI)
        roi_h, roi_w = roi_frame_gray.shape
        y_grid, x_grid = np.mgrid[step//2:roi_h:step, step//2:roi_w:step].reshape(2, -1).astype(int)
        
        for i in range(len(x_grid)):
            xi, yi = x_grid[i], y_grid[i]
            # Get flow at this point
            dx, dy = flow[yi, xi]
            
            # Only draw if motion is significant
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > motion_threshold:
                # Calculate start and end points in full frame coordinates
                start_x = x + xi
                start_y = y + yi
                end_x = int(start_x + dx)
                end_y = int(start_y + dy)
                
                # Draw blue arrow
                cv2.arrowedLine(flow_viz, (start_x, start_y), (end_x, end_y),
                               (0, 255, 0),  # Blue color (BGR format)
                               thickness=2,
                               tipLength=0.4)
        
        # Create image to save (without ROI box)
        img_to_save = flow_viz.copy()
        
        # Create display version with ROI rectangle
        img_display = flow_viz.copy()
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(img_display, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Scale for display and saving
        scaled_img_save = cv2.resize(img_to_save, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        scaled_img_display = cv2.resize(img_display, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        
        # Determine if this frame should be saved
        should_save = frame_count >= next_frame_to_save
        
        if should_save:
            # Save frame without ROI box
            if save_video:
                out.write(scaled_img_save)
                saved_count += 1
            
            # Update next frame to save
            next_frame_to_save += frame_interval
            
            # Show what's being saved (without ROI box)
            if save_video:
                cv2.imshow('Saved Output (Preview)', scaled_img_save)
        
        # Always display with ROI box
        cv2.imshow('Dense Optical Flow (Display)', scaled_img_display)
        
        # Check for ESC key
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            print("Stopped by user")
            break
        
        # Update for next iteration
        old_gray = frame_gray.copy()
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, saved {saved_count} frames...")
    
    # Cleanup
    cap.release()
    if save_video:
        out.release()
        print(f"Video saved to: {output_path}")
        print(f"Total frames processed: {frame_count}, saved: {saved_count}")
    cv2.destroyAllWindows()


def select_roi_interactive(video_path):
    """
    Interactive ROI selection tool.
    Returns: (x, y, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        return None
    
    print("Select ROI and press ENTER or SPACE")
    print("Cancel with ESC")
    
    # Select ROI
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    if roi[2] == 0 or roi[3] == 0:
        print("No ROI selected")
        return None
    
    print(f"Selected ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
    return roi


def main(args=None):
    video_path = "/home/ale/tutorials/computer-vision-deep-learning/code-python/training_misc/optical_flow/barreleye.mp4"
    
    # Define ROI (same as sparse optical flow)
    roi = (140, 0, 500, 280)
    
    # Or use interactive selection
    # roi = select_roi_interactive(video_path)
    # if roi is None:
    #     return
    
    start_dense_optical_flow(video_path, 
                            save_video=True,
                            output_path="/home/ale/tutorials/computer-vision-deep-learning/code-python/training_misc/optical_flow/barreleye_dense_optical_flow.mp4",
                            roi=roi)

    return

if __name__ == '__main__':
    main()