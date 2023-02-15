
#
# You Only Look Once (YOLO): an object detection algorithm using Deep Neural
# Networks by Joseph Redmond, Santosh Divvala, Ross Girshick, and Ali Farhadi.
# Redmond (grad student) and Farhadi (advisor) continued to improve upon the
# work and has created several new versions of the algorithm. We are going to
# be using Redmond's latest YOLOv3 pre-trained with Microsoft's COCO database.
#
# Redmond quit the Computer Vision field in early 2020 due to concerns with the
# application of his research. He was opposed to Facebook and Google using his
# work to invade people's privacy and the military who might weaponize it. He
# famously tweeeted:
#    I stopped doing CV research because I saw the impact my work was having. 
#    I loved the work but the military applications and privacy concerns 
#    eventually became impossible to ignore.
#
# Redmon was challenged to consider the many potential benefits of his
# technology and if these benefits might outweight the potential abuses. In
# reponse, Redmond hinted that his work had already been used to create weapons
# and that his funding was coming from people who were trying to kill people
# more efficiently. This is perhaps as good a time as any to pause and consider
# the potential applications of our technology in light of our commitment to
# the Great Commission and pursuit of goodness, beauty, and truth.
#
#   https://twitter.com/pjreddie/status/1230524770350817280?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1230524770350817280%7Ctwgr%5E75850dbadeab1a80880d2a69696a67c6500207ea%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fanalyticsindiamag.com%2Fyolo-creator-joe-redmon-computer-vision-research-ethical-concern%2F
#   https://www.deeplearning.ai/the-batch/code-no-evil/
#
# Since Redmond left the field, two new versions of his algorithm have been
# released, the most recent (v5) having some controversy of its own.
#   https://viso.ai/deep-learning/yolov5-controversy/
#
# Redmond is now a professional circus performer.
#

# Reference Articles:
# Object Detection 1 - https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# Object Detection 2 - https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# Object Detection 3 - https://dontrepeatyourself.org/post/object-detection-with-python-deep-learning-and-opencv/
# Blobs - https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
# Non Maxima Supression - https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/


import numpy as np
import time
import cv2
import sys
import os




# Adopted from https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088
def cv2_wait_for_window_to_close(window_title):
    """
    Blocks execution until the specified window has been closed. If the user
    presses the <ESC> key, the program will immediately quit.
    """

    value = 1.0
    while value >= 1.0:
        value = cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE)
        keyCode = cv2.waitKey(100)
        if (keyCode & 0xFF) == 27:
            print("Detected <ESC> Key... Quitting Program")
            cv2.destroyWindow(window_title)
            exit()




def load_yolo_deep_neural_network():
    """
    Loads the YOLOv3 object detection algorithm that has been trained with the
    Microsoft Common Objects in Context Dataset (COCO) to identify 80 objects.
    The three YOLOv3-COCO data files must be located in the same directory as
    this python script. See https://arxiv.org/abs/1405.0312.

    Returns a tuple containing 3 objects:
      0. Fully trained YOLO Deep Neurel Network Object classifier
      1. List of the DNN classifier's output layers by name
      2. List of the classification labels for the 80 known COCO objects
    """


    # Download weights: wget https://pjreddie.com/media/files/yolov3.weights
    # Download coco.names: wget https://github.com/pjreddie/darknet/blob/master/data/coco.names
    # Download yolov3.cfg: wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    labels_path = f'coco.names'
    config_path = f'yolov3.cfg'
    weights_path = f'yolov3.weights'

    with open(labels_path, 'r') as f:
        dnn_labels = [line.strip('\n') for line in f]

    dnn_object = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    dnn_layers = dnn_object.getUnconnectedOutLayersNames()

    return (dnn_object, dnn_layers, dnn_labels)




def detect_persons(img, dnn_object, obj_confidence, nms_threshold):
    """
    Detects COCO objects in an image with an OpenCV Deep Neural Network using
    Non-Maxima Supression to reduce the number of duplicate objects.

    Returns a list of detected objects, each defined as a tuple:
      0. COCO label, as an index number
      1. DNN confidence score
      2. Bounding box for the object
    """

    # Process raw image through the neural network to obtain potential objects
    #  => 1/255 is the scaling factor --> RGB value to a percentage
    #  => (224, 224) is the size of the output blob with smaller sizees being
    #     faster but potentially less accurate. The number came from this
    #     article by Adrian Rosebrock on PyImageSearch and produced fairly
    #     accurate results during some limited testing.

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), swapRB=True, crop=False)
    
    # Run the DNN object detection algorithm

    dnn_classifier, dnn_outputlayers = dnn_object
    dnn_classifier.setInput(blob)
    outputs = dnn_classifier.forward(dnn_outputlayers)
    flattened_outputs = [result for layer in outputs for result in layer]

    # We will identify objects by their location (the bounding box), the
    #   confidence score, and then the COCO label assigned by the DNN.
    # We need the image width x height to create the bounding boxes

    boxes = []
    scores = []
    labels = []
    img_h, img_w = img.shape[:2]

    # Each result is a nested list of all the detected objects. At the top
    #   level, we have a list of objects. Within each object, we are given a
    #   list containing the 4 coordinates of a bounding box followed by 80
    #   classification scores. There are 80 scores because our DNN was trained
    #   with 80 objects from the COCO dataset. We need to extract the bounding
    #   box and then identify the highest scoring object.
    # DNN bounding boxes identified by center, width, and height. We'll convert
    #   these to the upper-left coordinates of the box and the width & height.
    # Filter down to only the highest scoring people objects (label #0)

    for result in flattened_outputs:
        bbox = result[:4]
        all_scores = result[5:]
        best_label = np.argmax(all_scores)
        best_score = all_scores[best_label]
        
        if best_score > obj_confidence and best_label == 0:
            cx, cy, w, h = bbox * np.array([img_w, img_h, img_w, img_h])
            x = cx - w / 2
            y = cy - h / 2
            labels.append(best_label)
            scores.append(float(best_score))
            boxes.append([int(x), int(y), int(w), int(h)])

    # The DNN is likely to have identfied the same object multiple times, with
    #   each repeat found in a slightly different, overlapped, region of the
    #   image. We use the Non-Maxima Supression algorithm to detect redundant
    #   objects and return the best fitting bounding box from amongst all of
    #   the candidates.  

    best_idx = cv2.dnn.NMSBoxes(boxes, scores, obj_confidence, nms_threshold)
    if len(best_idx) > 0:
        objects = [(labels[i], scores[i], boxes[i]) for i in best_idx.flatten()]
    else:
        objects = []
    
    return objects




def process_image(img, dnn_object, confidence, threshold):
    """
    Detect persons in an image file using Deep Neural Network object detection
    algorithm. The image file will be downsized to fit within 640x480 before
    it is run through the DNN.

    Returns a new version of the image that has been overlayed with recangular
    bounding boxes and also a list of tuples identifying each object. The tuple
    contains the following three fields:
      0. COCO label, as an index number
      1. DNN confidence score
      2. Bounding box for the object 
    """

    # Resize the image to fit within 640x480 for easier viewing
    height, width = img.shape[:2]
    if width > 640 or height > 640:
        largest_dimension = max(height, width)
        scale_factor = 1 + largest_dimension // 640
        dimensions = (width // scale_factor, height // scale_factor)
        img = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
        height, width = img.shape[:2]

    # Detect all of the objects in this image
    objects = detect_persons(img, dnn_object, confidence, threshold)

    # Place a visual bounding box around each object detected in the image
    bgr_red = (0, 0, 255)
    for label, score, (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x+w, y+h), bgr_red, 2)
        #text = f"{label_names[label]}: {100*score:.1f}%"
        #cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, bgr_red, 2)

    return img, objects




if __name__ == '__main__':

    # Load a pre-trained Deep Neural Network to detect persons (and some 79 
    # other) contained in an image.

    dnn_classifier, dnn_layers, label_names = load_yolo_deep_neural_network()
    dnn_object = (dnn_classifier, dnn_layers)

    # Confidence is a Machine Learning classification score and it has to with
    #   the accuracy of the object detector--how likely is it that this object
    #   in the image is actually the given label
    # Threshold is a Non-Maxima Suppression value that helps us to minimize the
    #   number of duplicate objects detected in the image. Without it, we are
    #   likely to identify the same object multiple times. 

    confidence = 0.90
    threshold = 0.3

    # Command line parameters must be an image file or a dir containing images

    if len(sys.argv) < 2:
        print(f"Error: missing filename")
        print(f"Usage: python {sys.argv[0]} <file>")
        exit()
    file_arg = sys.argv[1]
    if not os.path.exists(file_arg):
        print(f"Error: '{file_arg}' does not exist")
        exit()
    if os.path.isfile(file_arg):
        image_files = [file_arg]
    else:
        image_files = [f"{file_arg}/{name}" for name in os.listdir(file_arg)]

    # Run through each image file one a time:
    #  1. Detect the person objects in the image
    #  2. Print the results to the console window
    #  3. Draw the image with bounding boxes in a GUI window
    #  4. Wait for the current GUI window to close
    # Unfortunately, cv2.moveWindow is not working on Chromebooks and so the
    # windows are being drawn in the center of the screen, which often means
    # the user has to drag them to a better location (annoying)

    for filename in image_files:
    
        time_start = time.time()

        img = cv2.imread(filename)
        img, objs = process_image(img, dnn_object, confidence, threshold)
        
        time_end = time.time()
        time_elapsed = time_end - time_start

        print(f'{filename}: detected {len(objs)} people objects in {time_elapsed:.3f}s')
        for label, score, (x, y, w, h) in objs:
            print(f'  => {label_names[label]}: {100*score:.1f} at ({x},{y})->({x+w},{y+h})')

        cv2.namedWindow(filename)
        cv2.imshow(filename, img)
        cv2.moveWindow(filename, 20, 20)
        cv2_wait_for_window_to_close(filename)
