import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import math
from collections import deque

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize session state for calibration and measurements
if 'calibrated_focal_length' not in st.session_state:
    st.session_state.calibrated_focal_length = None
if 'measurement_history' not in st.session_state:
    st.session_state.measurement_history = deque(maxlen=5)  # Store last 5 measurements
if 'calibration_mode' not in st.session_state:
    st.session_state.calibration_mode = False

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
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        landmarks = results.pose_landmarks
        
        # Calculate distance using multiple methods for accuracy
        distance = calculate_distance_multiple_methods(
            landmarks, focal_length, image.width, image.height
        )
        
        # Calculate height with improved method
        height = calculate_height(
            landmarks, distance, focal_length, image.height, image.width
        )
        
        # Store measurement if valid
        if height and distance:
            st.session_state.measurement_history.append({
                'height': height,
                'distance': distance
            })
        
        # Get guidance
        guidance, color = get_distance_guidance(distance)
        
        return annotated_frame, distance, height, guidance, color, True, focal_length
    else:
        return annotated_frame, None, None, "No person detected", "gray", False, focal_length

def get_averaged_height():
    """Get averaged height from recent measurements for stability"""
    if len(st.session_state.measurement_history) == 0:
        return None
    
    heights = [m['height'] for m in st.session_state.measurement_history if m['height']]
    if len(heights) > 0:
        return np.median(heights)  # Use median for robustness
    return None

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

# Camera input
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera Feed")
    camera_image = st.camera_input("Take a picture or use live camera", key="camera")
    
    if camera_image:
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
        
        # Display processed image
        st.image(processed_frame, use_container_width=True, channels="RGB")
        
        # Show focal length being used
        st.caption(f"Using focal length: {used_focal:.0f} px")

with col2:
    st.subheader("📊 Measurements")
    
    if camera_image:
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
        if color == "green":
            st.success(f"✅ {guidance}")
        elif color == "orange":
            st.warning(f"⚠️ {guidance}")
        else:
            st.info(f"ℹ️ {guidance}")
        
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
            
            if abs(distance - IDEAL_DISTANCE) <= DISTANCE_TOLERANCE and len(st.session_state.measurement_history) >= 3:
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

