import cv2
from picamera2 import Picamera2
import spidev
import RPi.GPIO as GPIO
import time
from twilio.rest import Client  # Twilio 라이브러리 추가
print(Clinet)

# Twilio 설정
TWILIO_ACCOUNT_SID = '사용자의 Twilio 계정 SID 입력'  # Twilio 계정 SID
TWILIO_AUTH_TOKEN = '사용자의 Twilio 인증 토큰 입력'    # Twilio 인증 토큰
TWILIO_PHONE_NUMBER = '사용자의 Twilio 발신 번호 입력'      # Twilio 발신 번호
TARGET_PHONE_NUMBER = '수신자 번호 입력'      # 수신자 번호

# Twilio 클라이언트 초기화
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# LED 및 경광등 설정
ledWhite = 18
alarmLight = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(ledWhite, GPIO.OUT)
GPIO.setup(alarmLight, GPIO.OUT)

# SPI 설정
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

def analogRead(ch):
    buf = [1, (8 + ch) << 4, 0]
    buf = spi.xfer2(buf)
    adcValue = ((buf[1] & 3) << 8) | buf[2]
    return adcValue

def send_sms(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=TARGET_PHONE_NUMBER
        )
        print(f"SMS 전송 성공: {message}")
    except Exception as e:
        print(f"SMS 전송 실패: {e}")

# 클래스 이름 설정
classNames = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
              15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
              35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
              42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
              59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
              70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
              79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
              88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def id_class_name(class_id, classes):
    return classes.get(class_id, "Unknown")

# Picamera2 초기화
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

def main():
    previous_detected_classes = set()
    try:
        model = cv2.dnn.readNetFromTensorflow(
            '/home/heejong/opencv/OpencvDnn/models/frozen_inference_graph.pb',
            '/home/heejong/opencv/OpencvDnn/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
        )

        object_detected = False
        dark_start_time = None

        while True:
            cdsValue = analogRead(0)
            print(f"CDS Value: {cdsValue}")

            GPIO.output(ledWhite, GPIO.HIGH if cdsValue < 500 else GPIO.LOW)

            image = picam2.capture_array()
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            image_height, image_width, _ = image.shape
            model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
            output = model.forward()

            detected_classes = set()
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .5:
                    class_id = int(detection[1])
                    class_name = id_class_name(class_id, classNames)
                    detected_classes.add(class_name)

            if object_detected:
                disappeared_objects = previous_detected_classes - detected_classes
                for obj in disappeared_objects:
                    warning_message = f"경고: '{obj}' 물체 사라짐!"
                    print(warning_message)
                    send_sms(warning_message)

            previous_detected_classes = detected_classes
            object_detected = bool(detected_classes)

            if cdsValue < 500 and len(detected_classes) == 0:
                if dark_start_time is None:
                    dark_start_time = time.time()
                elif time.time() - dark_start_time > 5:
                    GPIO.output(alarmLight, GPIO.HIGH)
                    dark_warning_message = "경고: 어두운 환경 5초 이상 유지됨!"
                    print(dark_warning_message)
                    send_sms(dark_warning_message)
            else:
                dark_start_time = None
                GPIO.output(alarmLight, GPIO.LOW)

            cv2.imshow('Camera', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        GPIO.cleanup()
        spi.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
