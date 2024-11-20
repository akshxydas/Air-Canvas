from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
import datetime
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BlipProcessor, BlipForConditionalGeneration


app = Flask(__name__)

# Directories for saving images and challenges
if not os.path.exists('static/images'):
    os.makedirs('static/images')

if not os.path.exists('static/challenges'):
    os.makedirs('static/challenges')

# Deques to hold points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
# Indices for managing drawing points
blue_index = 0
green_index = 0
red_index = 0
kernel = np.ones((5, 5), np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0
paintWindow = np.zeros((471, 636, 3)) + 255
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Stack to hold previous states for undo functionality
undo_stack = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_color', methods=['POST'])
def change_color():
    global colorIndex
    color = request.json['color']
    if color == 'clear':
        clear_canvas()
    elif color == 'blue':
        colorIndex = 0
    elif color == 'green':
        colorIndex = 1
    elif color == 'red':
        colorIndex = 2
    return jsonify(success=True)

@app.route('/save_image', methods=['POST'])
def save_image():
    global paintWindow
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    filepath = os.path.join('static/images', filename)
    cv2.imwrite(filepath, paintWindow)
    return jsonify(success=True, filename=filename)

@app.route('/delete_image', methods=['POST'])
def delete_image():
    filename = request.json['filename']
    filepath = os.path.join('static/images', filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify(success=True)
    return jsonify(success=False)

@app.route('/saved_images')
def saved_images():
    image_names = os.listdir('static/images')
    return render_template('saved_images.html', images=image_names)

@app.route('/challenges')
def challenges():
    challenge_images = os.listdir('static/challenges')
    return render_template('challenges.html', challenges=challenge_images)

@app.route('/submit_challenge', methods=['POST'])
def submit_challenge():
    challenge_image = request.json['challenge']
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_challenge.png'
    filepath = os.path.join('static/images', filename)
    cv2.imwrite(filepath, paintWindow)
    return jsonify(success=True, filename=filename)

@app.route('/start_drawing_challenge/<filename>')
def start_drawing_challenge(filename):
    return render_template('drawing_challenge.html', challenge_image=filename)

@app.route('/challenge_images/<filename>')
def challenge_images(filename):
    return send_from_directory('static/challenges', filename)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    filename = request.json['filename']
    filepath = os.path.join('static/images', filename)

    if os.path.exists(filepath):
        try:
            # Load the pre-trained TrOCR model and processor from Hugging Face
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

            # Open the image
            image = Image.open(filepath)

            inputs = processor(image, return_tensors="pt").pixel_values

            generated_ids = model.generate(inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return jsonify(success=True, text=generated_text)
        except Exception as e:
            return jsonify(success=False, message=f"TrOCR failed: {str(e)}")

    return jsonify(success=False, message="File not found")

# Initialize the image captioning model
captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/caption_image', methods=['POST'])
def caption_image():
    filename = request.json['filename']
    filepath = os.path.join('static/images', filename)

    if os.path.exists(filepath):
        try:
            # Open the image
            image = Image.open(filepath)

            # Preprocess the image and generate a caption
            inputs = captioning_processor(images=image, return_tensors="pt")
            out = captioning_model.generate(**inputs)
            caption = captioning_processor.decode(out[0], skip_special_tokens=True)

            return jsonify(success=True, caption=caption)
        except Exception as e:
            return jsonify(success=False, message=f"Image captioning failed: {str(e)}")

    return jsonify(success=False, message="File not found")


def clear_canvas():
    global bpoints, gpoints, rpoints, blue_index, green_index, red_index, paintWindow
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    blue_index = 0
    green_index = 0
    red_index = 0
    paintWindow[67:, :, :] = 255

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def is_five_finger_spread(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    distances = [
        calculate_distance(thumb_tip, wrist),
        calculate_distance(index_tip, wrist),
        calculate_distance(middle_tip, wrist),
        calculate_distance(ring_tip, wrist),
        calculate_distance(pinky_tip, wrist)
    ]

    return all(dist > 100 for dist in distances)


def generate_frames():
    global bpoints, gpoints, rpoints, blue_index, green_index, red_index, paintWindow, colorIndex, undo_stack
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.rectangle(frame, (40, 1), (150, 65), (255, 255, 255), 2)
        frame = cv2.rectangle(frame, (160, 1), (260, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (270, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (380, 1), (480, 65), (0, 0, 255), 2)
        cv2.putText(frame, "CLEAR", (70, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (190, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (290, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (410, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        result = hands.process(framergb)
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            if is_five_finger_spread(landmarks):
                erase_radius = 10
                for point in landmarks:
                    cv2.circle(frame, (point[0], point[1]), erase_radius, (255, 255, 255), -1)
                    cv2.circle(paintWindow, (point[0], point[1]), erase_radius, (255, 255, 255), -1)

                # Clear the points to stop drawing them again after erasing
                clear_canvas()


            else:
                fore_finger = (landmarks[8][0], landmarks[8][1])
                center = fore_finger
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, center, 3, (0, 255, 0), -1)

                if (thumb[1] - center[1] < 30):
                    bpoints.append(deque(maxlen=512))
                    blue_index += 1
                    gpoints.append(deque(maxlen=512))
                    green_index += 1
                    rpoints.append(deque(maxlen=512))
                    red_index += 1
                elif center[1] <= 65:
                    if 40 <= center[0] <= 140:
                        clear_canvas()
                    elif 160 <= center[0] <= 255:
                        colorIndex = 0
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2
                else:
                    # Save the current state for undo functionality
                    undo_stack.append({
                        'bpoints': [deque(maxlen=512) for _ in range(len(bpoints))],
                        'gpoints': [deque(maxlen=512) for _ in range(len(gpoints))],
                        'rpoints': [deque(maxlen=512) for _ in range(len(rpoints))],
                        'paintWindow': paintWindow.copy()
                    })

                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)

        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1

        points = [bpoints, gpoints, rpoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()