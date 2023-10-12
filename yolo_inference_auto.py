import math

import os
import time

from ultralytics import YOLO
import torch
import cv2


def load_images_path_from_folder(folder_path=None):
    image_names_list = []
    try:
        if os.path.isdir(folder_path):
            for images in os.listdir(folder_path):
                if images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg"):
                    image_names_list.append(str(images))
            if len(image_names_list) > 1:
                print("images paths loaded successfully")
                return sorted(image_names_list)
            else:
                print("no images found in folder: ", str(folder_path), " check folder again")
                raise Exception("empty image folder")
        else:
            raise Exception("folder path doesn't exists check path again")

    except Exception as loading_exception:
        print("error: exception in loading images from folder: ", loading_exception)


def detect_image_using_yolo(image):
    cords_list = []
    print("predicting image")
    s = time.process_time()
    # ------------------------------------------------------ prediction -----------------------------------------------
    results = model.predict(source=image,
                            show=False,
                            conf=0.5,
                            device=1,
                            save=False)
    t = time.process_time()
    print(f"prediction took {t - s} seconds")

    # <------------------------------------------ printing results ---------------------------------------------->
    # print(results)
    if len(results) >= 1:
        for cnt1, result in enumerate(results):
            # print("result: ", result)
            print(f"<------------------------- detection for image ------------------------------------------------>")
            # detection

            no_of_boxes = len(result.boxes)

            image_shape1 = list(result.orig_shape)
            # image = np.array(result.orig_img)
            print("no of boxes: ", no_of_boxes)
            print("image original size", image_shape1)

            text1 = str(f"no of coconuts: {no_of_boxes}")
            image = cv2.putText(image,
                                text1,
                                (100, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA)

            cords_list, class_id_list1 = [], []
            for box in result.boxes:
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                class_id = int(box.cls[0].item())
                class_name = result.names[box.cls[0].item()]
                # conf = box.conf[0].item()
                conf = round(box.conf[0].item(), 2)

                cords_list.append(cords)
                class_id_list1.append(class_id)

                # print results
                # print ("class name:", class_name, "Object type:", class_id)
                # print("Coordinates:", cords) # get box coordinates in (top, left, bottom, right) format
                # print("Probability:", conf)

            # saving detection image to train
            # print("cords_list", cords_list, "class_id_list1", class_id_list1)
    else:
        print("no results found")
    return cords_list, class_id_list1


# checking gpu was available
if torch.cuda.is_available():
    print("gpu is available")
processor_name = torch.cuda.get_device_name(0)
print("using ", processor_name, " to detect data")

# initializing the model
model_path = "D:/downloads/yolo_object_detection/yolo_custom_images/coconut_400.pt"
image_directory_path = os.getcwd() + "/images"

get_images_from_folder = True
cam_id = 1
print("loading model..........")
model = YOLO(model_path)
print("model loaded")
bounding_boxes_list = []

cropping = False
deleting = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


def diagonal_distance(px1=None, py1=None, px2=None, py2=None):
    dist = math.sqrt(((px1 - px2) ** 2) + ((py1 - py2) ** 2))
    print("diagonal_distance:", dist)
    return dist


def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, present_bounding_boxes_list
    t = 0

    if event == cv2.EVENT_LBUTTONDBLCLK:
        for each_cord1 in present_bounding_boxes_list:
            # get coordinates
            [x11, y11, x22, y22] = each_cord1
            if (x11 < x < x22) and (y11 < y < y22):
                cv2.rectangle(oriImage,
                              (x11, y11),
                              (x22, y22),
                              (100, 100, 100),
                              -1)
                present_bounding_boxes_list.remove(each_cord1)

    # if the left mouse button was DOWN
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start = x
        y_start = y
        x_end = x
        y_end = y
        cropping = True
    # Mouse is Moving

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end = x
            y_end = y
            # print(x_start, y_start, x_end, y_end, x, y)

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:

        x_end = x
        y_end = y

        # print("final values ->", x_start, y_start, x_end, y_end)
        if x_start > x_end:
            # swap both
            x_start, x_end = x_end, x_start
        if y_start > y_end:
            y_start, y_end = y_end, y_start

        # print("after sorted", x_start, y_start, x_end, y_end)

        cropping = False  # cropping is finished

        # displaying cropped image
        if diagonal_distance(x_start, y_start, x_end, y_end) > 20:
            present_bounding_boxes_list.append([x_start, y_start, x_end, y_end])

        print("bounding_boxes_list:", present_bounding_boxes_list)


print("<------------------------ loading images from folder ---------------------------------------------------->")
loaded_images_path = load_images_path_from_folder(image_directory_path)
no_of_images = len(loaded_images_path)

print("<------------  getting detections ----------------------------------------------------------------------->")
final_bounding_boxes_list = []
final_confidence_list = []
for cnt, im2 in enumerate(loaded_images_path):
    print(f"<--------------------------{cnt}/{no_of_images}-------------------------->", "\n")
    image2 = cv2.imread(image_directory_path + "/" + str(im2))
    bounding_boxes_list1, confidence_list1 = detect_image_using_yolo(image2)
    final_bounding_boxes_list.append(bounding_boxes_list1)
    final_confidence_list.append(confidence_list1)
print(
    "<------------  detections are collected ----------------------------------------------------------------------->")

edit = True
index1 = 0
while edit:
    if index1 <= 0:
        index1 = 0
    if index1 >= (no_of_images - 1):
        index1 = (no_of_images - 1)

    image_path = loaded_images_path[index1]
    image2 = cv2.imread(image_directory_path + "/" + str(image_path))
    bounding_boxes_list, confidence_list = final_bounding_boxes_list[index1], final_confidence_list[index1]

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    present_bounding_boxes_list = bounding_boxes_list

    while True:
        oriImage = image2.copy()

        # Drawing
        for each_cord in present_bounding_boxes_list:
            # get coordinates
            [x0, y0, x1, y1] = each_cord
            start_point1 = (int(x0), int(y0))
            end_point1 = (int(x1), int(y1))

            # display bounding box
            cv2.rectangle(oriImage,
                          start_point1,
                          end_point1,
                          (0, 255, 0),
                          2)
            if cropping:
                cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

            # display probability
            # cv2.putText(oriImage,
            #             str(each_conf),
            #             (x0 - 10, y0),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1,
            #             (0, 255, 0),
            #             2,
            #             cv2.LINE_AA)

        cv2.imshow("image", oriImage)

        # print("bounding_boxes_list", bounding_boxes_list)
        key = cv2.waitKey(1)
        if key == ord('a'):
            # replace a final list with a new list
            if present_bounding_boxes_list != bounding_boxes_list:
                final_bounding_boxes_list[index1] = present_bounding_boxes_list

            index1 -= 1
            break

        if key == ord('d'):
            # replace a final list with a new list
            if present_bounding_boxes_list != bounding_boxes_list:
                final_bounding_boxes_list[index1] = present_bounding_boxes_list

            index1 += 1
            break

        if key == 27:
            edit = False
            break

cv2.destroyAllWindows()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def image_detection_to_yolo_txt(image_shape, coordinates_list, image_path, class_id_list, output_path):
    # get image name
    basename = os.path.basename(image_path)
    image_name = os.path.splitext(basename)[0]

    print("text files save in ", output_path)

    text_file_path = output_path + "/classes.txt"
    if os.path.isfile(text_file_path):
        print("classes text file already exist")
    else:
        print("classes text file not exist copying from other folder")

    out_file = open(output_path + str('/') + image_name + '.txt', 'w')

    [height, width] = image_shape
    # print("image width ", width)
    # print("image height ", height)
    for coordinates, class_id in zip(coordinates_list, class_id_list):
        [x0, y0, x1, y1] = coordinates

        b = (float(x0),
             float(x1),
             float(y0),
             float(y1))
        bb = convert((width, height), b)
        # print("converted bounding box", bb)

        out_file.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# root_dir_path1 = "D:/downloads/images/yolo_data_set/val"
# image_path1 = "D:/downloads/images/yolo_data_set/val/bunch431.jpeg"
# output_path1 = "D:/downloads/images/yolo_data_set/val"
# convert_annotation(root_dir_path1, output_path1, image_path1)


# # <---------------------------------------parameters--------------------------------------------------->
# # represents the top left corner of image
# start_point = (0, 0)
# # represents the bottom right corner of the image
# end_point = (250, 250)
#
# # Green color in BGR
# color = (0, 0, 0)
# # Line thickness of 9 px
# thickness = 2
# # font

# font = cv2.FONT_HERSHEY_SIMPLEX
# # org

# org = (2, 50)
# # fontScale

# fontScale = 0.7
# # Blue color in BGR
# text_color = (0, 0, 0)
# # Line thickness of 2 px
# text_thickness = 2

# counter = 0

# fps = 0
# fps_avg_frame_count = 24
# start_time, end_time = 0, 0
# # <---------------------------------------------------------------------------------------------->
