import cv2

from models import YOLOMODEL

model = YOLOMODEL(
    weights_path='./weights/model_gun.pt',
)

guns = ['short_weapons','long_weapons', 'knife']
people = ['man_with_weapon', 'man_without_weapon']

def check_noise(frame, noise_threshold = 50, overexposed_threshold = 200):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray)[0]
    if mean_brightness < noise_threshold:
        print("Изображение зашумлено")
    else:
        print("Изображение не зашумлено")

    if mean_brightness > overexposed_threshold:
        print("Изображение засвечено")
    else:
        print("Изображение не засвечено")

def draw_bounding_box(frame, bboxes, labels, scores, keypoints):
    conf_people_with_guns = (set(guns) & set(labels)) and ((set(people) & set(labels))) and False
    if conf_people_with_guns:
        print("search gun and people")
        for bbox, label, score in zip(bboxes, labels, scores, keypoints):
            label_true = label
            if label in people:
                people_box = bbox
                xmin_people, ymin_people, xmax_people, ymax_people = map(int, people_box)
                for bbox, label, score in zip(bboxes, labels, scores):
                    if label in guns:
                        gun_box = bbox
                        xmin_guns, ymin_guns, xmax_guns, ymax_guns = map(int, gun_box)
                        if (xmin_people <= xmax_guns and xmax_people >= xmin_guns) and (
                                ymin_people <= ymax_guns and ymax_people >= ymin_guns):
                            label_true = 'man_with_weapon'
                            print("swap")
                            break
                        else:
                            label_true = 'man_without_weapon'
                            print("no swap")
                            break
            label = label_true
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', frame)
        cv2.resizeWindow('custom window', 700, 700)
        cv2.waitKey(1)

    # условие в IF Дорабатывается, менять все в else
    else:
        for bbox, label, score, kps in zip(bboxes, labels, scores, keypoints):
            if label in people:
                label = 'man_without_weapon'
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            #вывод ключевых точек на сервисе не будет!
            """"
            for kp in kps:
                for i in kp:
                    x, y = map(int, i)
                    cv2.circle(frame, (x, y), 3, (0, 0,255), -1)
            """
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', frame)
        cv2.resizeWindow('custom window', 700, 700)
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
