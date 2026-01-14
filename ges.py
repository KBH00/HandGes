import cv2
import time
from collections import deque

import mediapipe as mp

# -----------------------------
# Gesture definitions (5)
# -----------------------------
GESTURES = [
    "OPEN_PALM",   # all 5 fingers up
    "FIST",        # all down
    "THUMBS_UP",   # only thumb up
    "PEACE",       # index + middle up
    "POINT",       # only index up
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def _finger_up_states(hand_landmarks, handedness_label):
    """
    Returns dict of {thumb, index, middle, ring, pinky}: bool (up/down)
    Heuristic-based using landmark coordinates.
    """
    lm = hand_landmarks.landmark

    # Landmark indices (MediaPipe Hands)
    TH_TIP, TH_IP, TH_MCP = 4, 3, 2
    IN_TIP, IN_PIP = 8, 6
    MI_TIP, MI_PIP = 12, 10
    RI_TIP, RI_PIP = 16, 14
    PI_TIP, PI_PIP = 20, 18

    # For fingers except thumb: "up" if TIP is above PIP in image (y smaller)
    index_up = lm[IN_TIP].y < lm[IN_PIP].y
    middle_up = lm[MI_TIP].y < lm[MI_PIP].y
    ring_up = lm[RI_TIP].y < lm[RI_PIP].y
    pinky_up = lm[PI_TIP].y < lm[PI_PIP].y

    # Thumb is tricky: use handedness + x-direction AND y-direction as a combined heuristic.
    # - For a "thumbs up", thumb tip tends to be above IP (y smaller)
    thumb_vertical_up = lm[TH_TIP].y < lm[TH_IP].y

    # Also check thumb "openness" relative to MCP direction (x)
    # Right hand: thumb tip tends to be left of thumb MCP when extended outward (often),
    # Left hand: thumb tip tends to be right of thumb MCP when extended outward.
    if handedness_label == "Right":
        thumb_open = lm[TH_TIP].x < lm[TH_MCP].x
    else:  # "Left"
        thumb_open = lm[TH_TIP].x > lm[TH_MCP].x

    # Consider thumb "up" if it's vertically up OR clearly open (helps stability)
    thumb_up = thumb_vertical_up or thumb_open

    return {
        "thumb": thumb_up,
        "index": index_up,
        "middle": middle_up,
        "ring": ring_up,
        "pinky": pinky_up,
    }


def classify_gesture(states):
    """
    states: dict thumb/index/middle/ring/pinky -> bool
    Returns one of GESTURES or "UNKNOWN"
    """
    t = states["thumb"]
    i = states["index"]
    m = states["middle"]
    r = states["ring"]
    p = states["pinky"]

    up_count = sum([t, i, m, r, p])

    # 1) FIST
    if up_count == 0:
        return "FIST"

    # 2) OPEN_PALM
    if up_count == 5:
        return "OPEN_PALM"

    # 3) THUMBS_UP (only thumb up)
    if t and not (i or m or r or p):
        return "THUMBS_UP"

    # 4) PEACE (index + middle only)
    if i and m and not (t or r or p):
        return "PEACE"

    # 5) POINT (index only)
    if i and not (t or m or r or p):
        return "POINT"

    return "UNKNOWN"


def stable_label(label_queue):
    """
    Simple temporal smoothing: choose the most frequent label in recent frames.
    """
    if not label_queue:
        return "UNKNOWN"
    freq = {}
    for x in label_queue:
        freq[x] = freq.get(x, 0) + 1
    return max(freq.items(), key=lambda kv: kv[1])[0]


def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    # Smoothing buffer
    recent = deque(maxlen=10)
    last_print = ""
    last_print_time = 0.0

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            gesture = "NO_HAND"
            handedness = ""

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_landmarks = result.multi_hand_landmarks[0]
                handedness = result.multi_handedness[0].classification[0].label  # "Left"/"Right"

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(thickness=2),
                )

                states = _finger_up_states(hand_landmarks, handedness)
                gesture = classify_gesture(states)

                recent.append(gesture)
                gesture = stable_label(recent)

                # Optional: show finger states on screen
                states_text = f"T:{int(states['thumb'])} I:{int(states['index'])} M:{int(states['middle'])} R:{int(states['ring'])} P:{int(states['pinky'])}"
                cv2.putText(frame, states_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                recent.clear()

            # Overlay
            title = f"Hand: {handedness or '-'}  Gesture: {gesture}"
            cv2.putText(frame, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Console print (rate-limited)
            now = time.time()
            if gesture != last_print and (now - last_print_time) > 0.2:
                print(title)
                last_print = gesture
                last_print_time = now

            cv2.imshow("MediaPipe Hand Gesture (5-class)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=0)
