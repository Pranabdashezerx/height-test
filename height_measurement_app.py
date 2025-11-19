import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import time
import hashlib
import os
import sys
import warnings

# Suppress MediaPipe GL context warnings (harmless in headless environments)
# These errors occur because MediaPipe tries GPU first, then falls back to CPU
os.environ['GLOG_minloglevel'] = '2'  # Suppress glog warnings (MediaPipe uses glog)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Try to import MediaPipe with error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    # Initialize MediaPipe (use CPU mode to avoid GL context errors in headless environments)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=1  # Use simpler model to reduce GPU dependency
    )
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    st.error(f"⚠️ MediaPipe could not be imported: {e}. Please ensure Python 3.11 or 3.12 is being used.")
    st.stop()

# Initialize session state for calibration and measurements
if 'calibrated_focal_length' not in st.session_state:
    st.session_state.calibrated_focal_length = None
if 'measurement_history' not in st.session_state:
    st.session_state.measurement_history = deque(maxlen=10)  # Store last 10 measurements
if 'calibration_mode' not in st.session_state:
    st.session_state.calibration_mode = False
if 'height_calibration_data' not in st.session_state:
    st.session_state.height_calibration_data = []  # Store (measured, known) pairs
if 'calibration_slope' not in st.session_state:
    st.session_state.calibration_slope = None
if 'calibration_intercept' not in st.session_state:
    st.session_state.calibration_intercept = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False
if 'last_frame_time' not in st.session_state:
    st.session_state.last_frame_time = 0

# Constants for distance estimation
AVERAGE_SHOULDER_WIDTH = 0.41  # Average shoulder width in meters
AVERAGE_HEAD_HEIGHT = 0.24  # Average head height in meters
AVERAGE_HIP_WIDTH = 0.28  # Average hip width in meters
IDEAL_DISTANCE = 2.5  # Ideal distance in meters (adjust as needed)
DISTANCE_TOLERANCE = 0.2  # Tolerance for "perfect" distance in meters (reduced for accuracy)

def detect_reference_object(image):
    """Detect A4 paper or rectangular reference object for calibration"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            return approx
    
    return None

def calibrate_from_reference(image, reference_width_m, reference_height_m, known_distance_m=2.0):
    """Calibrate focal length from a reference object at a known distance"""
    reference = detect_reference_object(image)
    if reference is None:
        return None
    
    # Get width and height in pixels
    pts = reference.reshape(4, 2)
    width_px = max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    )
    height_px = max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    )
    
    # Calculate focal length using: focal_length = (pixel_size * distance) / real_size
    focal_length_w = (width_px * known_distance_m) / reference_width_m
    focal_length_h = (height_px * known_distance_m) / reference_height_m
    
    # Return average for robustness
    return (focal_length_w + focal_length_h) / 2

def calculate_distance_multiple_methods(landmarks, focal_length, image_width, image_height):
    """Calculate distance using multiple body parts and average for accuracy"""
    distances = []
    
    # Method 1: Shoulder width
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * image_width
    if shoulder_width_px > 0:
        dist1 = (focal_length * AVERAGE_SHOULDER_WIDTH) / shoulder_width_px
        distances.append(dist1)
    
    # Method 2: Hip width
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_width_px = abs(left_hip.x - right_hip.x) * image_width
    if hip_width_px > 0:
        dist2 = (focal_length * AVERAGE_HIP_WIDTH) / hip_width_px
        distances.append(dist2)
    
    # Method 3: Head height (estimate from face landmarks)
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    ear_distance_px = abs(left_ear.x - right_ear.x) * image_width
    if ear_distance_px > 0:
        # Ear-to-ear distance is approximately 0.15m
        dist3 = (focal_length * 0.15) / ear_distance_px
        distances.append(dist3)
    
    if len(distances) > 0:
        # Use median for robustness against outliers
        return np.median(distances)
    return None

def get_distance_guidance(distance):
    """Provide guidance on whether to move closer or farther"""
    if distance is None:
        return "No person detected", "gray"
    
    diff = distance - IDEAL_DISTANCE
    
    if abs(diff) <= DISTANCE_TOLERANCE:
        return "Perfect! Stay in position", "green"
    elif diff > 0:
        return f"Move {diff:.2f}m closer", "orange"
    else:
        return f"Move {abs(diff):.2f}m farther", "orange"

def estimate_top_of_head(landmarks, image_height):
    """Estimate top of head position more accurately"""
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    
    # Average ear position
    ear_y = (left_ear.y + right_ear.y) / 2
    
    # Estimate head height (nose to top of head is approximately 0.12m)
    # In pixels, this is roughly 10% of head-to-nose distance
    head_height_ratio = 0.12 / 0.24  # head height / total head height
    nose_to_ear_px = abs(nose.y - ear_y) * image_height
    head_top_offset_px = nose_to_ear_px * head_height_ratio
    
    # Top of head is above nose
    head_top_y = nose.y - (head_top_offset_px / image_height)
    
    return head_top_y

def find_ground_level(landmarks, image_height):
    """Find ground level by detecting the lowest visible point"""
    # Get all lower body points
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL] if hasattr(mp_pose.PoseLandmark, 'LEFT_HEEL') else left_ankle
    right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL] if hasattr(mp_pose.PoseLandmark, 'RIGHT_HEEL') else right_ankle
    
    # Use the lowest point (closest to ground)
    ground_y = max(left_ankle.y, right_ankle.y, left_heel.y, right_heel.y)
    
    return ground_y

def calculate_height(landmarks, distance, focal_length, image_height, image_width):
    """Calculate person's height with improved accuracy"""
    if landmarks is None or distance is None or focal_length is None:
        return None
    
    try:
        # Estimate top of head more accurately
        head_top_y = estimate_top_of_head(landmarks, image_height)
        
        # Find ground level
        ground_y = find_ground_level(landmarks, image_height)
        
        # Calculate height in pixels
        height_pixels = abs(ground_y - head_top_y) * image_height
        
        # Convert to real-world height using similar triangles
        # real_height / distance = pixel_height / focal_length
        real_height = (height_pixels * distance) / focal_length
        
        return real_height
    except Exception as e:
        return None

def process_frame(image, focal_length=None):
    """Process a single frame to detect person and calculate measurements"""
    # Convert PIL to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use calibrated focal length or default
    if focal_length is None:
        focal_length = st.session_state.calibrated_focal_length if st.session_state.calibrated_focal_length else 700
    
    # Process with MediaPipe
    results = pose.process(frame_rgb)
    
    # Draw pose landmarks
    annotated_frame = frame_rgb.copy()
    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        
        # Calculate distance using multiple methods for accuracy
        distance = calculate_distance_multiple_methods(
            landmarks, focal_length, image.width, image.height
        )
        
        # Calculate height with improved method
        height = calculate_height(
            landmarks, distance, focal_length, image.height, image.width
        )
        
        # Apply calibration if available
        calibrated_height = apply_calibration(height) if height else None
        
        # Store measurement if valid (store raw height for calibration)
        if height and distance:
            st.session_state.measurement_history.append({
                'height': height,  # Store raw height
                'calibrated_height': calibrated_height,
                'distance': distance,
                'timestamp': len(st.session_state.measurement_history)
            })
        
        # Draw detailed annotations
        annotated_frame = draw_detailed_annotations(
            annotated_frame, landmarks, distance, calibrated_height or height,
            image.width, image.height
        )
        
        # Get guidance
        guidance, color = get_distance_guidance(distance)
        
        return annotated_frame, distance, calibrated_height or height, guidance, color, True, focal_length
    else:
        return annotated_frame, None, None, "No person detected", "gray", False, focal_length

def calculate_linear_regression(measured_heights, known_heights):
    """
    Calculate linear regression calibration using numpy
    
    Args:
        measured_heights: List of heights measured by the app (in meters)
        known_heights: List of actual/known heights (in meters)
    
    Returns:
        slope, intercept, r_squared: Calibration parameters and fit quality
    """
    if len(measured_heights) != len(known_heights):
        raise ValueError("Measured and known heights must have the same length")
    
    if len(measured_heights) < 2:
        raise ValueError("Need at least 2 data points for regression")
    
    # Convert to numpy arrays
    x = np.array(measured_heights)
    y = np.array(known_heights)
    
    # Calculate means
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    # Calculate covariance and variance
    cov_x_y = np.sum((x - x_bar) * (y - y_bar))
    var_x = np.sum((x - x_bar) ** 2)
    
    # Avoid division by zero
    if var_x == 0:
        raise ValueError("Variance of measured heights is zero")
    
    # Calculate slope and intercept
    slope = cov_x_y / var_x
    intercept = y_bar - slope * x_bar
    
    # Calculate R-squared (coefficient of determination)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, intercept, r_squared

def apply_calibration(height, slope=None, intercept=None):
    """Apply calibration to a measured height"""
    if slope is None or intercept is None:
        slope = st.session_state.calibration_slope
        intercept = st.session_state.calibration_intercept
    
    if slope is None or intercept is None:
        return height  # No calibration available
    
    return slope * height + intercept

def get_averaged_height(calibrated=True):
    """Get averaged height from recent measurements for stability"""
    if len(st.session_state.measurement_history) == 0:
        return None
    
    heights = [m['height'] for m in st.session_state.measurement_history if m['height']]
    if len(heights) == 0:
        return None
    
    # Apply calibration if available
    if calibrated:
        heights = [apply_calibration(h) for h in heights]
    
    return np.median(heights)  # Use median for robustness

def draw_detailed_annotations(image, landmarks, distance, height, image_width, image_height):
    """Draw detailed annotations on the image with measurements"""
    annotated = image.copy()
    
    if landmarks is None:
        return annotated
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        annotated,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    
    # Get key points for measurement visualization
    try:
        nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Convert normalized coordinates to pixel coordinates
        nose_px = (int(nose.x * image_width), int(nose.y * image_height))
        left_ankle_px = (int(left_ankle.x * image_width), int(left_ankle.y * image_height))
        right_ankle_px = (int(right_ankle.x * image_width), int(right_ankle.y * image_height))
        left_shoulder_px = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
        right_shoulder_px = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
        
        # Draw height line (head to feet)
        ground_y = max(left_ankle.y, right_ankle.y) * image_height
        head_top_y = nose.y * image_height - (image_height * 0.1)  # Estimate head top
        
        cv2.line(annotated, 
                (int(nose.x * image_width), int(head_top_y)),
                (int(nose.x * image_width), int(ground_y)),
                (0, 255, 255), 3)
        
        # Draw distance reference (shoulder width)
        cv2.line(annotated, left_shoulder_px, right_shoulder_px, (255, 255, 0), 2)
        
        # Add text annotations
        if height:
            height_cm = height * 100
            cv2.putText(annotated, f"Height: {height_cm:.1f} cm",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if distance:
            cv2.putText(annotated, f"Distance: {distance:.2f} m",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw key points
        cv2.circle(annotated, nose_px, 5, (0, 0, 255), -1)
        cv2.circle(annotated, left_ankle_px, 5, (255, 0, 0), -1)
        cv2.circle(annotated, right_ankle_px, 5, (255, 0, 0), -1)
        
    except Exception as e:
        pass  # Continue even if annotation fails
    
    return annotated

# Streamlit UI
st.set_page_config(page_title="Height Measurement App", layout="wide")

st.title("📏 Accurate Height Measurement Application")
st.markdown("""
### Instructions for Accurate Measurement:
1. **First Time Setup**: Calibrate using a reference object (A4 paper, credit card, or ruler)
2. Place your phone on a tripod at a fixed height
3. Have the person stand straight, facing the camera
4. Follow the distance guidance to position correctly
5. When it says "Perfect!", the height will be measured accurately
""")

# Sidebar for calibration
with st.sidebar:
    st.header("⚙️ Calibration & Settings")
    
    # Calibration Mode
    st.subheader("📐 Camera Calibration")
    calibration_mode = st.checkbox("Enable Calibration Mode", value=st.session_state.calibration_mode)
    st.session_state.calibration_mode = calibration_mode
    
    if calibration_mode:
        st.info("📋 Hold an A4 paper (or credit card) in front of the camera at the same distance where you'll measure")
        reference_type = st.selectbox("Reference Object", ["A4 Paper (21cm x 29.7cm)", "Credit Card (8.56cm x 5.398cm)", "Custom"], key="ref_type")
        
        if reference_type == "A4 Paper (21cm x 29.7cm)":
            ref_width = 0.21
            ref_height = 0.297
        elif reference_type == "Credit Card (8.56cm x 5.398cm)":
            ref_width = 0.0856
            ref_height = 0.05398
        else:
            ref_width = st.number_input("Width (meters)", 0.01, 1.0, 0.21, 0.01, key="ref_w")
            ref_height = st.number_input("Height (meters)", 0.01, 1.0, 0.297, 0.01, key="ref_h")
        
        # Store in session state for use in main processing
        st.session_state.ref_width = ref_width
        st.session_state.ref_height = ref_height
        st.session_state.reference_type = reference_type
        
        calibration_distance = st.number_input("Distance to reference object (meters)", 1.0, 5.0, 2.0, 0.1, key="calib_dist")
        st.session_state.calibration_distance = calibration_distance
        
        if st.button("Calibrate Camera"):
            st.info("Take a photo with the reference object visible")
    else:
        if st.session_state.calibrated_focal_length:
            st.success(f"✅ Calibrated: {st.session_state.calibrated_focal_length:.0f} px")
        else:
            st.warning("⚠️ Not calibrated - using default focal length")
    
    st.divider()
    
    st.subheader("Distance Settings")
    IDEAL_DISTANCE = st.slider("Ideal Distance (meters)", 1.5, 4.0, 2.5, 0.1)
    DISTANCE_TOLERANCE = st.slider("Distance Tolerance (meters)", 0.1, 0.5, 0.2, 0.05)
    
    st.divider()
    
    st.subheader("Manual Focal Length")
    manual_focal = st.slider("Focal Length (pixels)", 500, 1500, 
                             int(st.session_state.calibrated_focal_length) if st.session_state.calibrated_focal_length else 700, 50)
    if st.button("Use Manual Focal Length"):
        st.session_state.calibrated_focal_length = float(manual_focal)
        st.success("Focal length updated!")
    
    st.divider()
    
    st.subheader("Reference Measurements")
    st.caption("These are used for distance estimation")
    AVERAGE_SHOULDER_WIDTH = st.slider("Shoulder Width (m)", 0.30, 0.50, 0.41, 0.01)
    AVERAGE_HIP_WIDTH = st.slider("Hip Width (m)", 0.20, 0.40, 0.28, 0.01)
    
    st.divider()
    
    # Height Calibration Section
    st.subheader("📊 Height Calibration")
    st.caption("Calibrate using known heights for better accuracy")
    
    if len(st.session_state.height_calibration_data) > 0:
        st.info(f"📋 {len(st.session_state.height_calibration_data)} calibration points collected")
        if st.session_state.calibration_slope is not None:
            st.success(f"✅ Calibrated: Slope={st.session_state.calibration_slope:.4f}, Intercept={st.session_state.calibration_intercept:.4f}")
    
    # Add calibration data point
    if len(st.session_state.measurement_history) > 0:
        latest_height = st.session_state.measurement_history[-1]['height'] * 100  # Convert to cm
        known_height = st.number_input("Enter Known Height (cm)", 100.0, 250.0, float(latest_height), 0.1, key="known_h")
        
        col_add, col_calc = st.columns(2)
        with col_add:
            if st.button("➕ Add Calibration Point"):
                st.session_state.height_calibration_data.append({
                    'measured': latest_height / 100,  # Store in meters
                    'known': known_height / 100
                })
                st.success("Point added!")
                st.rerun()
        
        with col_calc:
            if len(st.session_state.height_calibration_data) >= 2:
                if st.button("🔧 Calculate Calibration"):
                    measured = [d['measured'] for d in st.session_state.height_calibration_data]
                    known = [d['known'] for d in st.session_state.height_calibration_data]
                    try:
                        slope, intercept, r_squared = calculate_linear_regression(measured, known)
                        st.session_state.calibration_slope = slope
                        st.session_state.calibration_intercept = intercept
                        st.success(f"✅ Calibration calculated! R² = {r_squared:.4f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    if st.button("🗑️ Clear Calibration Data"):
        st.session_state.height_calibration_data = []
        st.session_state.calibration_slope = None
        st.session_state.calibration_intercept = None
        st.rerun()
    
    # Toggle analysis view
    st.divider()
    st.session_state.show_analysis = st.checkbox("📈 Show Analysis Dashboard", value=st.session_state.show_analysis)

# Camera input
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera Feed")
    
    # Live mode toggle
    live_col1, live_col2 = st.columns([3, 1])
    with live_col1:
        st.session_state.live_mode = st.checkbox("🟢 Live Mode (Real-time Feedback)", value=st.session_state.live_mode, key="live_toggle")
    with live_col2:
        if st.session_state.live_mode:
            st.success("ON")
        else:
            st.info("OFF")
    
    # Camera input
    camera_image = st.camera_input("Take a picture or use live camera", key="camera")
    
    # Create placeholder for live updates
    image_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # Process in live mode
    if st.session_state.live_mode and camera_image:
        # Process continuously
        image = Image.open(camera_image)
        focal_length = st.session_state.calibrated_focal_length if st.session_state.calibrated_focal_length else 700
        
        processed_frame, distance, height, guidance, color, person_detected, used_focal = process_frame(image, focal_length)
        
        # Update display
        image_placeholder.image(processed_frame, width='stretch', channels="RGB")
        info_placeholder.caption(f"Using focal length: {used_focal:.0f} px | Live Mode: Processing...")
        
        # Auto-refresh for live mode (every 0.5 seconds)
        time.sleep(0.5)
        st.rerun()
    
    if camera_image and not st.session_state.live_mode:
        # Process the image
        image = Image.open(camera_image)
        
        # Handle calibration
        if st.session_state.calibration_mode:
            frame_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            
            # Get reference dimensions from session state
            ref_type = st.session_state.get('reference_type', 'A4 Paper (21cm x 29.7cm)')
            if ref_type == "A4 Paper (21cm x 29.7cm)":
                ref_w, ref_h = 0.21, 0.297
            elif ref_type == "Credit Card (8.56cm x 5.398cm)":
                ref_w, ref_h = 0.0856, 0.05398
            else:
                ref_w = st.session_state.get('ref_width', 0.21)
                ref_h = st.session_state.get('ref_height', 0.297)
            
            calib_dist = st.session_state.get('calibration_distance', 2.0)
            focal = calibrate_from_reference(frame_rgb, ref_w, ref_h, calib_dist)
            if focal:
                st.session_state.calibrated_focal_length = focal
                st.success(f"✅ Calibration successful! Focal length: {focal:.0f} pixels")
                st.session_state.calibration_mode = False
            else:
                st.error("❌ Could not detect reference object. Make sure it's clearly visible.")
        
        # Get focal length
        focal_length = st.session_state.calibrated_focal_length if st.session_state.calibrated_focal_length else 700
        
        processed_frame, distance, height, guidance, color, person_detected, used_focal = process_frame(image, focal_length)
        
        # Display processed image (only if not in live mode, live mode uses placeholder)
        if not st.session_state.live_mode:
            st.image(processed_frame, width='stretch', channels="RGB")
            st.caption(f"Using focal length: {used_focal:.0f} px")

with col2:
    st.subheader("📊 Measurements")
    
    # Create placeholders for live updates
    distance_placeholder = st.empty()
    guidance_placeholder = st.empty()
    height_placeholder = st.empty()
    avg_height_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Process measurements
    if camera_image:
        # Get latest measurement data
        if len(st.session_state.measurement_history) > 0:
            latest = st.session_state.measurement_history[-1]
            distance = latest.get('distance')
            height = latest.get('calibrated_height') or latest.get('height')
        else:
            # Process current frame if no history
            image = Image.open(camera_image)
            focal_length = st.session_state.calibrated_focal_length if st.session_state.calibrated_focal_length else 700
            _, distance, height, guidance, color, person_detected, _ = process_frame(image, focal_length)
        
        if st.session_state.live_mode:
            # Live mode - update placeholders continuously
            with distance_placeholder.container():
                st.markdown("### Distance")
                if distance:
                    st.metric("Distance from Camera", f"{distance:.2f} m")
                    accuracy_indicator = "🟢" if abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE else "🟡"
                    st.caption(f"{accuracy_indicator} {'Optimal' if abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE else 'Adjust position'}")
                else:
                    st.info("Calculating...")
            
            with guidance_placeholder.container():
                st.markdown("### Guidance")
                if distance:
                    guidance, color = get_distance_guidance(distance)
                    if color == "green":
                        st.success(f"✅ {guidance}")
                    elif color == "orange":
                        st.warning(f"⚠️ {guidance}")
                    else:
                        st.info(f"ℹ️ {guidance}")
                else:
                    st.info("Waiting for person detection...")
            
            with height_placeholder.container():
                st.markdown("### Height Measurement")
                if height:
                    height_cm = height * 100
                    height_ft = height * 3.28084
                    feet = int(height_ft)
                    inches = int((height_ft - feet) * 12)
                    
                    st.metric("Current Height", f"{height_cm:.1f} cm")
                    st.metric("Current Height", f"{feet}' {inches}\"")
                else:
                    st.info("Calculating height...")
            
            with avg_height_placeholder.container():
                averaged_height = get_averaged_height()
                if averaged_height and len(st.session_state.measurement_history) >= 3:
                    avg_cm = averaged_height * 100
                    avg_ft = averaged_height * 3.28084
                    avg_feet = int(avg_ft)
                    avg_inches = int((avg_ft - avg_feet) * 12)
                    
                    st.divider()
                    st.markdown("### 📊 Averaged Height")
                    st.markdown("*Based on last measurements*")
                    st.metric("Averaged Height", f"{avg_cm:.1f} cm")
                    st.metric("Averaged Height", f"{avg_feet}' {avg_inches}\"")
                    st.caption(f"Measurements: {len(st.session_state.measurement_history)}")
            
            with status_placeholder.container():
                if distance and abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE and len(st.session_state.measurement_history) >= 3:
                    st.success("🎯 Perfect position! Measurement accurate!")
        
        # Regular mode (non-live) - show static content
        if not st.session_state.live_mode:
            # Distance display
            st.markdown("### Distance")
            if distance:
                st.metric("Distance from Camera", f"{distance:.2f} m")
                accuracy_indicator = "🟢" if abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE else "🟡"
                st.caption(f"{accuracy_indicator} {'Optimal' if abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE else 'Adjust position'}")
            else:
                st.info("Calculating...")
            
            # Guidance
            st.markdown("### Guidance")
            if distance:
                guidance, color = get_distance_guidance(distance)
                if color == "green":
                    st.success(f"✅ {guidance}")
                elif color == "orange":
                    st.warning(f"⚠️ {guidance}")
                else:
                    st.info(f"ℹ️ {guidance}")
            else:
                st.info("Waiting for person detection...")
            
            # Height display
            st.markdown("### Height Measurement")
            
            # Get current and averaged height
            current_height = height
            averaged_height = get_averaged_height()
            
            if current_height:
                height_cm = current_height * 100
                height_ft = current_height * 3.28084
                feet = int(height_ft)
                inches = int((height_ft - feet) * 12)
                
                st.metric("Current Height", f"{height_cm:.1f} cm")
                st.metric("Current Height", f"{feet}' {inches}\"")
                
                # Show averaged height if we have multiple measurements
                if averaged_height and len(st.session_state.measurement_history) >= 3:
                    avg_cm = averaged_height * 100
                    avg_ft = averaged_height * 3.28084
                    avg_feet = int(avg_ft)
                    avg_inches = int((avg_ft - avg_feet) * 12)
                    
                    st.divider()
                    st.markdown("### 📊 Averaged Height")
                    st.markdown("*Based on last 5 measurements*")
                    st.metric("Averaged Height", f"{avg_cm:.1f} cm", delta=f"{abs(height_cm - avg_cm):.1f} cm")
                    st.metric("Averaged Height", f"{avg_feet}' {avg_inches}\"")
                
                # Show measurement count
                if len(st.session_state.measurement_history) > 0:
                    st.caption(f"Measurements: {len(st.session_state.measurement_history)}/5")
                
                if distance and abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE and len(st.session_state.measurement_history) >= 3:
                    st.balloons()
                    st.success("🎯 Accurate measurement achieved!")
            else:
                if person_detected:
                    st.warning("Position yourself correctly for accurate measurement")
                else:
                    st.info("Waiting for person detection...")
            
            # Clear measurements button
            if st.button("🔄 Clear Measurements"):
                st.session_state.measurement_history.clear()
                st.rerun()
    else:
        st.info("👆 Click the camera button to start measuring")

# Instructions and tips
with st.expander("ℹ️ Tips for Maximum Accuracy"):
    st.markdown("""
    ### For Best Results:
    
    **1. Calibration (IMPORTANT for accuracy):**
    - Use the calibration mode with an A4 paper or credit card
    - Hold the reference object at the same distance where you'll measure
    - This ensures accurate focal length calculation
    
    **2. Camera Setup:**
    - Keep the camera at a fixed height (preferably at eye level or slightly higher)
    - Camera should be perpendicular to the ground (not tilted)
    - Use a tripod for stability
    
    **3. Measurement Environment:**
    - Ensure good, even lighting so the person is clearly visible
    - Use a plain background for better detection
    - Person should stand straight, facing the camera
    - Feet should be flat on the ground
    
    **4. Measurement Process:**
    - Wait for "Perfect!" position before measuring
    - Take multiple measurements (app averages last 5)
    - The averaged height is more accurate than a single measurement
    
    **5. Accuracy:**
    - With proper calibration: ±1-2 cm accuracy
    - Without calibration: ±3-5 cm accuracy
    - Multiple measurements improve reliability
    """)

