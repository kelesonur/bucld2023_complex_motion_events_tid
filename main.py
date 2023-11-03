import os
import cv2
import xml.etree.cElementTree as et


def extract_frames(record_rate, info_folder, image_folder):

    if not os.path.isdir(info_folder):
        print("Video folder could not be found:", info_folder)
        return

    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    info = os.listdir(info_folder)

    info.sort()

    print(info)

    for k in info:

        # Read eaf and mp4 together
        if k[-4:] == ".eaf" and (k[:-4] + ".mp4") in info:

            # file = open(info_folder + "/" + k, encoding="utf8")
            mp4_name = info_folder + "/" + k[:-4] + ".mp4"

            # Skip if personal folder is not empty
            personal_folder = image_folder + "/" + k[:-4]
            if not os.path.isdir(personal_folder):
                os.mkdir(personal_folder)
            elif len(os.listdir(personal_folder)) > 0:
                continue

            print(k)

            # Parse eaf file
            tree = et.parse(info_folder + "/" + k)
            root = tree.getroot()

            # print(myroot.tag)
            time_slots = dict()
            labels = []

            for item_1 in root:
                # print(item_1.tag)
                if item_1.tag == "TIER":
                    labels.append([])
                for item_2 in item_1:
                    if item_2.tag == "TIME_SLOT":
                        # print(item_2.tag, item_2.attrib)
                        time_slots[item_2.attrib["TIME_SLOT_ID"]] = int(item_2.attrib["TIME_VALUE"])
                    elif item_2.tag == "ANNOTATION":
                        for item_3 in item_2:
                            for item_4 in item_3:
                                # print("a", item_3.tag, item_3.attrib, item_3.text)
                                # print("b", item_4.tag, item_4.attrib, item_4.text)
                                labels[-1].append([item_4.text, time_slots[item_3.attrib["TIME_SLOT_REF1"]], time_slots[item_3.attrib["TIME_SLOT_REF2"]]])

            # print(time_slots)
            # print(labels)
            # for p in labels:
            #     print(p)

            # Make video ready
            vidcap = cv2.VideoCapture(mp4_name)
            total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            print("Total Frames:", total_frames, "FPS:", fps, "Screenshot in every", int(record_rate * fps), "frame")

            for p in labels[3]:

                tier_1 = ""
                tier_2 = ""
                # print(p)
                for t in labels[1]:
                    if t[1] <= p[1] <= t[2] and t[1] <= p[2] <= t[2]:
                        tier_1 = t[0]
                        break
                for t in labels[2]:
                    if t[1] <= p[1] <= t[2] and t[1] <= p[2] <= t[2]:
                        tier_2 = t[0]
                        break
                print(p[0], tier_1, tier_2, p[1], p[2])
                if tier_1 == "" or tier_2 == "":
                    print("Label is missing:", p)

                # Read video
                start_frame = int(p[1] / 1000 * fps)
                end_frame = int(p[2] / 1000 * fps)
                print(start_frame, end_frame)

                counter = 1
                for f in range(start_frame, end_frame, int(record_rate * fps)):
                    vidcap = cv2.VideoCapture(mp4_name)
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    success, image = vidcap.read()
                    if not success:
                        print("Frame failed in:", k, p, f)
                    else:
                        image_path = str(start_frame) + "_" + str(end_frame) + "," + str(counter) + "," + p[0].replace(">", "-") + "," + tier_1 + "," + tier_2
                        cv2.imwrite(personal_folder + "/" + image_path + ".png", image)
                        counter = counter + 1


record_rate = 3/25
info_folder = "videos_and_labels"

image_folder = "people_images"
extract_frames(record_rate, info_folder, image_folder)

