import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for, session
import threading
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize MediaPipe Pose and Hands solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

POSE_LANDMARK_INDICES = [13, 14]  # Left Elbow, Right Elbow

HAND_LANDMARKS_LEFT1 = [5, 6, 7, 8]
HAND_LANDMARKS_LEFT2 = [5, 6, 7, 8, 9, 10, 11, 12]
HAND_LANDMARKS_LEFT3 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HAND_LANDMARKS_LEFT4 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
HAND_LANDMARKS_LEFT5 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

HAND_LANDMARKS_RIGHT1 = [5, 6, 7, 8]
HAND_LANDMARKS_RIGHT2 = [5, 6, 7, 8, 9, 10, 11, 12]
HAND_LANDMARKS_RIGHT3 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HAND_LANDMARKS_RIGHT4 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
HAND_LANDMARKS_RIGHT5 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Global variables for video capture and threading
cap = None
pose = None
hands = None
output_frame = None
lock = threading.Lock()
running = False

# Global variable to store current detected label
current_label = "not found"
label_lock = threading.Lock()

def check_finger_open(hand_landmarks, finger_indices):
    for tip_idx in finger_indices:
        lower_idx = tip_idx - 2
        tip = hand_landmarks.landmark[tip_idx]
        lower = hand_landmarks.landmark[lower_idx]
        distance = ((tip.x - lower.x) ** 2 + (tip.y - lower.y) ** 2) ** 0.5
        if distance < 0.03:
            return False
    return True

def video_capture_thread():
    global cap, pose, hands, output_frame, running

    cap = cv2.VideoCapture(0)  # Use default webcam 0
    if not cap.isOpened():
        print("Cannot open webcam")
        running = False
        return

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb_frame)
        results_hands = hands.process(rgb_frame)

        # Flags
        left_elbow_detected = False
        right_elbow_detected = False
        left_hand_left1_detected = False
        left_hand_left2_detected = False
        left_hand_left3_detected = False
        left_hand_left4_detected = False
        left_hand_left5_detected = False
        right_hand_right1_detected = False
        right_hand_right2_detected = False
        right_hand_right3_detected = False
        right_hand_right4_detected = False
        right_hand_right5_detected = False  
        left_hand_leftbonus_detected = False
        right_hand_rightbonus_detected = False

        # Check pose landmarks
        if results_pose.pose_landmarks:
            for index in POSE_LANDMARK_INDICES:
                landmark = results_pose.pose_landmarks.landmark[index]
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, frame_width, frame_height)
                if landmark_px:
                    cv2.circle(frame, landmark_px, 10, (0, 255, 0), cv2.FILLED)
                    if index == 13:
                        left_elbow_detected = True
                    if index == 14:
                        right_elbow_detected = True

        # Check hand landmarks
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'

                if label == 'Left':
                    if check_finger_open(hand_landmarks, HAND_LANDMARKS_LEFT1):
                        left_hand_left1_detected = True
                        for idx in HAND_LANDMARKS_LEFT1:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (255, 0, 0), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [9, 10, 11, 12]) and left_elbow_detected:
                        left_hand_left2_detected = True
                        for idx in HAND_LANDMARKS_LEFT2:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (0, 255, 255), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [13, 14, 15, 16]) and left_elbow_detected:
                        left_hand_left3_detected = True
                        for idx in HAND_LANDMARKS_LEFT3:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (255, 0, 255), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [17, 18, 19, 20]) and left_elbow_detected:
                        left_hand_left4_detected = True
                        for idx in HAND_LANDMARKS_LEFT4:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (0, 165, 255), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) and left_elbow_detected:
                        left_hand_left5_detected = True
                        for idx in HAND_LANDMARKS_LEFT5:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (128, 0, 128), cv2.FILLED)
                    if check_finger_open(hand_landmarks, [1, 2, 3, 4]) and left_elbow_detected:
                        left_hand_leftbonus_detected = True
                        for idx in [1, 2, 3, 4]:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (0, 255, 128), cv2.FILLED)

                elif label == 'Right':
                    if check_finger_open(hand_landmarks, HAND_LANDMARKS_RIGHT1):
                        right_hand_right1_detected = True
                        for idx in HAND_LANDMARKS_RIGHT1:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (0, 0, 255), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [9, 10, 11, 12]) and right_elbow_detected:
                        right_hand_right2_detected = True
                        for idx in HAND_LANDMARKS_RIGHT2:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (255, 255, 0), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [13, 14, 15, 16]) and right_elbow_detected:
                        right_hand_right3_detected = True
                        for idx in HAND_LANDMARKS_RIGHT3:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (0, 255, 0), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [17, 18, 19, 20]) and right_elbow_detected:
                        right_hand_right4_detected = True
                        for idx in HAND_LANDMARKS_RIGHT4:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (255, 165, 0), cv2.FILLED)

                    if check_finger_open(hand_landmarks, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) and right_elbow_detected:
                        right_hand_right5_detected = True
                        for idx in HAND_LANDMARKS_RIGHT5:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (0, 128, 255), cv2.FILLED)
                    if check_finger_open(hand_landmarks, [1, 2, 3, 4]) and right_elbow_detected:
                        right_hand_rightbonus_detected = True
                        for idx in [1, 2, 3, 4]:
                            cv2.circle(frame, mp_drawing._normalized_to_pixel_coordinates(
                                hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y, frame_width, frame_height), 8, (255, 165, 0), cv2.FILLED)

        # Determine label based on the detected hand gestures
        if (left_elbow_detected and left_hand_leftbonus_detected and
            not (left_hand_left1_detected or left_hand_left2_detected or
                 left_hand_left3_detected or left_hand_left4_detected or
                 left_hand_left5_detected or right_hand_right1_detected or
                 right_hand_right2_detected or right_hand_right3_detected or 
                 right_hand_right4_detected or right_hand_right5_detected)):
            label_text = "rightbonus"
        elif (right_elbow_detected and right_hand_rightbonus_detected and
            not (right_hand_right1_detected or right_hand_right2_detected or
                 right_hand_right3_detected or right_hand_right4_detected or
                 right_hand_right5_detected or left_hand_left1_detected or 
                 left_hand_left2_detected or left_hand_left3_detected or 
                 left_hand_left4_detected or left_hand_left5_detected)):
            label_text = "leftbonus"
        elif left_elbow_detected and left_hand_left5_detected:
            label_text = "left5"
        elif left_elbow_detected and left_hand_left4_detected:
            label_text = "left4"
        elif left_elbow_detected and left_hand_left3_detected:
            label_text = "left3"
        elif left_elbow_detected and left_hand_left2_detected:
            label_text = "left2"
        elif left_elbow_detected and left_hand_left1_detected:
            label_text = "left1"
        elif right_elbow_detected and right_hand_right5_detected:
            label_text = "right5"
        elif right_elbow_detected and right_hand_right4_detected:
            label_text = "right4"
        elif right_elbow_detected and right_hand_right3_detected:
            label_text = "right3"
        elif right_elbow_detected and right_hand_right2_detected:
            label_text = "right2"
        elif right_elbow_detected and right_hand_right1_detected:
            label_text = "right1"
        else:
            label_text = "not found"

        # Update global current_label with thread safety
        with label_lock:
            global current_label
            current_label = label_text

        # Display the label
        label_position = (10, 30)
        cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Acquire lock to set the output frame
        with lock:
            output_frame = frame.copy()

    # Release resources
    cap.release()
    pose.close()
    hands.close()

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Simple hardcoded authentication
        if username == 'admin' and password == 'password':
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    stop_video()
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/start')
def start_video():
    global running
    if 'username' not in session:
        return redirect(url_for('login'))
    if not running:
        running = True
        thread = threading.Thread(target=video_capture_thread)
        thread.start()
    return ('', 204)

@app.route('/stop')
def stop_video():
    global running, cap
    if 'username' not in session:
        return redirect(url_for('login'))
    running = False
    # Release capture if exists
    if cap is not None:
        cap.release()
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_label')
def current_label_route():
    if 'username' not in session:
        return redirect(url_for('login'))
    with label_lock:
        label = current_label
    return {"label": label}

@app.route('/saved_data')
def saved_data():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('saved_data.html')

if __name__ == '__main__':
    print("Running app.run()...")  # Add this line
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("app.run() finished.") 

