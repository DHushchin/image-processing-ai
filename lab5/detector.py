from imageai.Detection import ObjectDetection, VideoObjectDetection
import os


def photo_detector(input_path, output_path, model_path):
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(model_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"])
        
    print()
    

def video_detector(input_path, output_path, model_path):
    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(model_path)
    detector.loadModel()

    video_path = detector.detectObjectsFromVideo(input_file_path=input_path,
                                                 output_file_path=output_path,
                                                 frames_per_second=20,
                                                 log_progress=True)

    print(video_path)


def main():
    model_path = os.getcwd() + '/model/tiny-yolov3.pt'
    
    print('Test 1: test.jpg')
    input_path = os.getcwd() + '/input/test.jpg'
    output_path = os.getcwd() + '/output/test.jpg'
    photo_detector(input_path, output_path, model_path)
    
    print('Test 2: fruits.jpg')
    input_path = os.getcwd() + '/input/fruits.jpg'
    output_path = os.getcwd() + '/output/fruits.jpg'
    photo_detector(input_path, output_path, model_path)
    
    print('Test 3: test.mp4')
    input_path = os.getcwd() + '/input/people.mp4'
    output_path = os.getcwd() + '/output/people'
    video_detector(input_path, output_path, model_path)
    
    
if __name__ == "__main__":
    main()

