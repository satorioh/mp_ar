import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "../model/hand_landmarker.task"
camera = 0
width = 1280
height = 720


def init_detector():
    base_options = BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                    result_callback=print_result)
    return HandLandmarker.create_from_options(options)


def init_camera():
    cap = cv2.VideoCapture(camera)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    return cap


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))


def main():
    cap = init_camera()
    detector = init_detector()
    timestamp = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        cv2.imshow('MediaPipe Hand Landmarker', mp_image.numpy_view())

        timestamp += 1
        detector.detect_async(mp_image, timestamp)

        if cv2.waitKey(5) & 0xFF == 27:
            print("Closing Camera Stream")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
