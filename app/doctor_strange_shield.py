import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "../model/hand_landmarker.task"
SHIELD_1 = cv2.imread('magic_circles/magic_circle_ccw.png', -1)
SHIELD_2 = cv2.imread('magic_circles/magic_circle_cw.png', -1)
DEG = 0
CAMERA_DEVICE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


class ShieldModule:
    def __init__(self):
        self.result = None

    def init_detector(self):
        base_options = BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.GPU)
        options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                        num_hands=2,
                                        result_callback=self.print_result)
        return HandLandmarker.create_from_options(options)

    def init_camera(self):
        cap = cv2.VideoCapture(CAMERA_DEVICE)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        return cap

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        hand_landmarks_list_len = len(hand_landmarks_list)
        for idx in range(hand_landmarks_list_len):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def draw_line(self, img, p1, p2, size=5):
        cv2.line(img, p1, p2, (50, 50, 255), size)
        cv2.line(img, p1, p2, (255, 255, 255), round(size / 2))

    def position_data(self, lmlist):
        global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip
        wrist = (lmlist[0][0], lmlist[0][1])
        thumb_tip = (lmlist[4][0], lmlist[4][1])
        index_mcp = (lmlist[5][0], lmlist[5][1])
        index_tip = (lmlist[8][0], lmlist[8][1])
        midle_mcp = (lmlist[9][0], lmlist[9][1])
        midle_tip = (lmlist[12][0], lmlist[12][1])
        ring_tip = (lmlist[16][0], lmlist[16][1])
        pinky_tip = (lmlist[20][0], lmlist[20][1])

    def calculate_distance(self, p1, p2):
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)

    def calculate_ratio(self, img_w, img_h):
        for hand in self.result.hand_landmarks:
            lm_list = []
            for idx, lm in enumerate(hand):
                coor_x, coor_y = int(lm.x * img_w), int(lm.y * img_h)
                lm_list.append([coor_x, coor_y])
            self.position_data(lm_list)
            palm = self.calculate_distance(wrist, index_mcp)
            distance = self.calculate_distance(index_tip, pinky_tip)
            ratio = distance / palm
            return ratio

    def main(self):
        cap = self.init_camera()
        detector = self.init_detector()
        timestamp = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(frame, 1)
            image_for_detect = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))

            timestamp += 1
            detector.detect_async(image_for_detect, timestamp)

            if self.result is not None:
                h, w, c = image.shape
                ratio = self.calculate_ratio(w, h)
                print(ratio)
                if ratio and (0.5 < ratio < 1.5):
                    self.draw_line(image, wrist, thumb_tip)
                    self.draw_line(image, wrist, index_tip)
                    self.draw_line(image, wrist, midle_tip)
                    self.draw_line(image, wrist, ring_tip)
                    self.draw_line(image, wrist, pinky_tip)
                    self.draw_line(image, thumb_tip, index_tip)
                    self.draw_line(image, thumb_tip, midle_tip)
                    self.draw_line(image, thumb_tip, ring_tip)
                    self.draw_line(image, thumb_tip, pinky_tip)
                cv2.imshow('show frame', image)
            else:
                cv2.imshow('show frame', image)

            if cv2.waitKey(5) & 0xFF == 27:
                print("Closing Camera Stream")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    shield_module = ShieldModule()
    shield_module.main()
