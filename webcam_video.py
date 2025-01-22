import cv2
import numpy as np
import time

# 모델 및 구성 파일 경로 설정
config_path = "model/yolov3-tiny.cfg"
weights_path = "model/yolov3-tiny.weights"
classes_path = "model/coco.names"

# 클래스 이름 로드
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 네트워크 로드
net = cv2.dnn.readNet(weights_path, config_path)

# GPU 사용 가능 여부 확인 및 설정
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("GPU를 사용합니다.")
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("GPU가 없습니다, CPU를 사용합니다.")

# 사용자 입력으로 웹캠 또는 비디오 파일 선택
source = input("웹캠을 사용하려면 'webcam', 비디오 파일을 사용하려면 'video'를 입력하세요: ").strip().lower()

if source == 'video':
    video_path = "data/morning.mp4"
    cap = cv2.VideoCapture(video_path)
    print(f"비디오 파일 {video_path}을(를) 재생합니다.")
else:
    cap = cv2.VideoCapture(0)
    print("웹캠을 사용합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 출력 레이어 이름 가져오기
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 객체 감지 실행
    start_time = time.time()
    outputs = net.forward(output_layers)
    end_time = time.time()

    # 감지된 객체 정보 저장
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # 비최대 억제 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과 시각화
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS 계산 및 표시
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 결과 표시
    cv2.imshow("Tiny YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
