import cv2

from models import YOLOMODEL

#Объявдение нашей модели YOLOv8 medium
model = YOLOMODEL(
    weights_path='./weights/model_gun.pt',
)

guns = ['short_weapons','long_weapons', 'knife']
people = ['man_with_weapon', 'man_without_weapon']

#Функция для анализа зашумленности и засвета камеры
def check_noise(frame, noise_threshold = 50, overexposed_threshold = 200):
    gray = frame.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray)[0]
    if mean_brightness < noise_threshold:
        print("Изображение зашумлено")
    else:
        print("Изображение не зашумлено")

    if mean_brightness > overexposed_threshold:
        print("Изображение засвечено")
    else:
        print("Изображение не засвечено")

#Вывод Bbox и аннотации на Frontend
def draw_bounding_box(frame, bboxes, labels, scores, keypoints):
    conf_people_with_guns = (set(guns) & set(labels)) and ((set(people) & set(labels)))
    if conf_people_with_guns:
        for bbox, label, score in zip(bboxes, labels, scores):
            label_true = label
            if label in people:
                people_box = bbox
                xmin_people, ymin_people, xmax_people, ymax_people = map(int, people_box) # Алгоритм проверки, точно ли человек вооруженный или нет
                for bbox, label, score in zip(bboxes, labels, scores):
                    if label in guns:
                        gun_box = bbox
                        xmin_guns, ymin_guns, xmax_guns, ymax_guns = map(int, gun_box)
                        if (xmin_people <= xmax_guns and xmax_people >= xmin_guns) and (
                                ymin_people <= ymax_guns and ymax_people >= ymin_guns):
                            label_true = 'man_with_weapon'
                            break
                        else:
                            label_true = 'man_without_weapon'
                            break
            label = label_true
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', frame)
        cv2.resizeWindow('custom window', 1000, 1000)
        cv2.waitKey(2)

    # условие в IF Дорабатывается, менять все в else
    else:
        for bbox, label, score in zip(bboxes, labels, scores):
            if label in people:
                label = 'man_without_weapon'
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            #вывод ключевых точек на сервисе не будет!
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', frame)
        cv2.resizeWindow('custom window', 1000, 1000)
        cv2.waitKey(2)


def main():
    source = 'video.mp4' # вставляй сюда rtsp ссылку
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        check_noise(frame)

        bboxes, labels, scores, keypoints = model.predict(frame)
        draw_bounding_box(frame, bboxes, labels, scores, keypoints)

if __name__ == '__main__':
    main()