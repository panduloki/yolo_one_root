import math

import os
import sys
import time

from ultralytics import YOLO
import torch
import cv2

train_text_file_path = os.getcwd() + "/train/train.txt"


def get_images_from_train_text_file(train_text_file_path1):
    image_paths = []
    if os.path.isfile(train_text_file_path1):
        print("train.txt file already exist")
        with open(train_text_file_path1, "r") as file1:
            for line in file1:
                # print(line.rstrip())
                image_paths.append(line.rstrip())
        file1.close()

    else:
        print("classes text file not exist ")
    return image_paths


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
image_directory_path = os.getcwd() + "/train/images"

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
    global x_start, y_start, x_end, y_end, cropping, present_bounding_boxes_list, oriImage
    (h, w, _) = oriImage.shape

    # limiting mouse x,y in boundaries
    if x <= 0:
        x = 0
    if y <= 0:
        y = 0

    if x >= w:
        x = w
    if y >= h:
        y = h

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
# print("loaded image_path_list", loaded_images_path)

text_detected_image = get_images_from_train_text_file(train_text_file_path)
print("image_names_list after detecting from train.txt", text_detected_image)

# deduct already yolo detected images
yolo_not_detected_images = []
yolo_detected_image_paths = []
for each_path in loaded_images_path:
    if each_path in text_detected_image:
        yolo_detected_image_paths.append(each_path)
    else:
        yolo_not_detected_images.append(each_path)
print(
    f"after segregating yolo detected paths: {yolo_detected_image_paths} yolo undetected image paths"
    f" {yolo_not_detected_images}")

no_of_images = len(yolo_not_detected_images)
if no_of_images > 0:
    print("<------------  getting detections ----------------------------------------------------------------------->")
    final_bounding_boxes_list = []
    final_confidence_list = []
    for cnt, im2 in enumerate(yolo_not_detected_images):
        print(f"<--------------------------{cnt}/{no_of_images - 1}-------------------------->", "\n")
        image2 = cv2.imread(image_directory_path + "/" + str(im2))
        bounding_boxes_list1, confidence_list1 = detect_image_using_yolo(image2)
        final_bounding_boxes_list.append(bounding_boxes_list1)
        final_confidence_list.append(confidence_list1)

    print("<------------  detections are collected ------------------------------------------------------------------->")
    completed_image_path_list = []
    edit = True
    index1 = 0


    while edit:
        # print(f"index:{index1} / no_of_images:{no_of_images}")

        if index1 <= 0:
            index1 = 0
        if index1 >= no_of_images:
            index1 = (no_of_images - 1)

        image_path = yolo_not_detected_images[index1]
        image2 = cv2.imread(image_directory_path + "/" + str(image_path))
        bounding_boxes_list, confidence_list = final_bounding_boxes_list[index1], final_confidence_list[index1]

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)

        present_bounding_boxes_list = bounding_boxes_list

        # editing image
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

                # images which are done stored in a finished list
                if image_path not in completed_image_path_list:
                    completed_image_path_list.append(image_path)

                index1 -= 1
                break

            if key == ord('d'):
                # replace a final list with a new list
                if present_bounding_boxes_list != bounding_boxes_list:
                    final_bounding_boxes_list[index1] = present_bounding_boxes_list

                # images which are done stored in a finished list
                if image_path not in completed_image_path_list:
                    completed_image_path_list.append(image_path)

                index1 += 1
                break

            if key == 27:
                edit = False
                # if only one image left, add an image path to a completed image path list
                if no_of_images == 1:
                    # images which are done stored in a finished list
                    if image_path not in completed_image_path_list:
                        completed_image_path_list.append(image_path)

                break
    cv2.destroyAllWindows()

    # after completed image store image names in train.txt
    print("completed_image_path_list", completed_image_path_list)
    try:
        train_txt_file = open(train_text_file_path, 'w+')
        completed_image_path_list += yolo_detected_image_paths
        for each_name in completed_image_path_list:
            train_txt_file.write(each_name + '\n')
        train_txt_file.close()

    except Exception as text_exception:
        print("error: exception in loading train.txt  folder: ", text_exception)

    print("<------------  editing  completed  ------------------------------------------------------------------->")


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


    def image_detection_to_yolo_txt(image_shape, coordinates_list, image_path1, output_path):
        # get image name
        basename = os.path.basename(image_path1)
        image_name = os.path.splitext(basename)[0]

        print("text files save in ", output_path)

        text_file_path = output_path + "/classes.txt"
        if os.path.isfile(text_file_path):
            print("classes text file already exist")
        else:
            print("classes text file not exist copying from other folder")

        out_file = open(output_path + str('/') + image_name + '.txt', 'w')

        (height, width, _) = image_shape
        # print("image width ", width)
        # print("image height ", height)
        for coordinates in coordinates_list:
            [x0, y0, x1, y1] = coordinates

            b = (float(x0),
                 float(x1),
                 float(y0),
                 float(y1))
            bb = convert((width, height), b)
            # print("converted bounding box", bb)

            out_file.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()


    # class_id_list2 = [0] * no_of_images
    # print("class_id_list2", class_id_list2)

    for id1 in range(no_of_images):
        print(f"<--------------------------{id1}/{no_of_images - 1}-------------------------->", "\n")
        image_path3 = loaded_images_path[id1]
        image3 = cv2.imread(image_directory_path + "/" + str(image_path3))
        img_shape = image3.shape
        output_path2 = os.getcwd() + "/train/labels"
        image_detection_to_yolo_txt(img_shape, final_bounding_boxes_list[id1], image_path3, output_path2)

    print("<------------  images yolo detection saved in labels    ------------------------------------------------->")

sys.exit()

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
