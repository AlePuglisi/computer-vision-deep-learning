import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt 


def start_lk_optical_flow(video_path, save_video=False, output_path="optical_flow_output.mp4", roi=None):
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
    
    # Target FPS for output (we'll process every frame and save at target fps)
    target_fps = 30
    
    # Set up video writer if saving
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Scale dimensions for output
        out_width = int(width * 0.4)
        out_height = int(height * 0.4)
        # Use target FPS for output
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (out_width, out_height))
        
        if not out.isOpened():
            print("Error: Could not open video writer")
            cap.release()
            return
    
    # Feature detection parameters
    feature_params = dict(maxCorners=100,
                         qualityLevel=0.1,
                         minDistance=10,
                         blockSize=10)
    
    # Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(10, 10), 
                    maxLevel=1,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create random colors for feature points
    color = np.random.randint(0, 255, (100, 3))
    
    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Create mask for feature detection (only within ROI)
    feature_mask = np.zeros_like(old_gray)
    feature_mask[y:y+h, x:x+w] = 255
    
    # Detect initial features only within ROI
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=feature_mask, **feature_params)
    
    if p0 is None:
        print("Error: No features detected in ROI")
        cap.release()
        if save_video:
            out.release()
        return
    
    print(f"Detected {len(p0)} initial features in ROI")
    
    # Create mask for drawing
    drawing_mask = np.zeros_like(old_frame)
    
    frame_count = 0
    saved_count = 0
    clear_interval = 30  # Clear trails every 30 frames
    
    # Calculate which frames to save based on fps ratio
    frame_interval = fps / target_fps
    next_frame_to_save = 0
    
    print(f"Processing video at {fps} fps, saving at {target_fps} fps (interval: {frame_interval:.2f})...")
    print("Press 'Esc' to stop early")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow for every frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None and st is not None:
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            # Draw tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b = int(a), int(b)
                c, d = int(c), int(d)
                
                drawing_mask = cv2.line(drawing_mask, (a, b), (c, d), color[i % len(color)].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 8, color[i % len(color)].tolist(), -1)
            
            # Update for next iteration
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        # Clear mask periodically and re-detect features in ROI
        if frame_count > 0 and frame_count % clear_interval == 0:
            drawing_mask = np.zeros_like(frame)
            # Re-detect features only within ROI
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_mask, **feature_params)
            if p0 is None:
                print(f"Warning: No features detected at frame {frame_count}")
                # Keep old points if detection fails
                p0 = good_new.reshape(-1, 1, 2) if 'good_new' in locals() else None
        
        # Combine frame and mask (this is what will be saved)
        img_to_save = cv2.add(frame, drawing_mask)
        
        # Create display version with ROI rectangle
        img_display = img_to_save.copy()
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
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
        cv2.imshow('LK Optical Flow (Display)', scaled_img_display)
        
        # Check for ESC key (use small wait time for smooth playback)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            print("Stopped by user")
            break
        
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
    # Run with or without saving
    roi = (140, 0, 500, 280)
    start_lk_optical_flow(video_path, save_video=True, 
                          output_path="/home/ale/tutorials/computer-vision-deep-learning/code-python/training_misc/optical_flow/barreleye_sparse_optical_flow.mp4",
                          roi=roi)

    return

if __name__ == '__main__':
    main()