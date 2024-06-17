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
CAMERA_DEVICE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


class HandLandmarkerModule:
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
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        return cap

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        print('hand landmarker result: {}'.format(result))
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

    def main(self):
        cap = self.init_camera()
        detector = self.init_detector()
        timestamp = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

            timestamp += 1
            detector.detect_async(mp_image, timestamp)

            if self.result is not None:
                annotated_image = self.draw_landmarks_on_image(frame, self.result)
                cv2.imshow('show annotated image', annotated_image)
            else:
                cv2.imshow('show frame', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                print("Closing Camera Stream")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    hand_landmarker_module = HandLandmarkerModule()
    hand_landmarker_module.main()
