import pickle
import cv2

from utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))

with open(f'{current_dir}/name_encodings.pickle', 'rb') as f:
    name_encodings = pickle.load(f)

image = cv2.imread(f'{current_dir}/examples/2.jpg')
# image = cv2.imread(f'{current_dir}/examples/personal/2.jpeg')
encodings = face_encodings(image)

names = []

for encoding in encodings:
    matches = {}

    for name, known_encodings in name_encodings.items():
        matches[name] = compare_faces(known_encodings, encoding)
    print(matches)
    if all(count == 0 for count in matches.values()):
        names.append('Unknown')
    else:
        names.append(max(matches, key=matches.get))


for rect, name in zip(face_rects(image), names):
    cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
    cv2.putText(image, name, (rect.left(), rect.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
