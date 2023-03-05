import os
import pickle

from utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))
image_paths = get_image_paths(f'{current_dir}/dataset', ['Arnold_Schwarzenegger', 'personal'])

name_encodings = {}

i = 0

for image_path in image_paths:
    image = cv2.imread(image_path)
    encodings = face_encodings(image)
    name = image_path.split(os.path.sep)[-2]
    e = name_encodings.get(name, [])
    e.extend(encodings)
    name_encodings[name] = e
    i += 1
    print(f'Processed {i} from {len(image_paths)} paths')


with open(f'{current_dir}/name_encodings.pickle', 'wb') as f:
    pickle.dump(name_encodings, f)
