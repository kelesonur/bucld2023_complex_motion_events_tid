import json
import os

import numpy as np

people_native = [3, 13, 14, 18, 20, 22, 26, 27, 33, 37]
people_late = [4, 16, 21, 36, 1, 10, 12, 19, 31, 32]


def calculate_body_movement_and_angles(image_folder):
    result_list = []

    info = os.listdir(image_folder)

    info.sort()

    print(info)

    np.set_printoptions(precision=3, suppress=True, linewidth=300)

    for k in info:

        print(k)

        json_folder = image_folder + "/" + k + "/" + "json"

        if not os.path.isdir(json_folder):
            print("Json file is not found:", k)
            continue

        json_list = os.listdir(json_folder)
        json_list.sort()

        json_lister = []
        for t in json_list:
            json_lister.append(t.split(","))

        groups = {}
        for l in json_lister:
            groups.setdefault(l[0], []).append(l)

        json_list_sorted = list(groups.values())
        for tt in json_list_sorted:

            points_left = np.zeros([len(tt), 6, 3])
            points_right = np.zeros([len(tt), 6, 3])

            json_number = 0
            for t in tt:
                json_name = ""
                for n in t:
                    json_name = json_name + n + ","
                json_name = json_name[:-1]
                with open(json_folder + "/" + json_name, 'r') as file:
                    data = json.load(file)

                    # Left Side
                    index = 0
                    for p in [0, 1, 5, 6, 7]:
                        for z in range(3):
                            points_left[json_number][index][z] = data["people"][0]["pose_keypoints_2d"][p * 3 + z]
                        index = index + 1

                    hand_sum = []
                    for p in range(21):
                        hand_position = [0, 0, 0]
                        for z in range(3):
                            hand_position[z] = data["people"][0]["hand_left_keypoints_2d"][p * 3 + z]
                        if 0.0 in hand_position:
                            continue
                        else:
                            hand_sum.append(hand_position)
                    hand_sum = np.array(hand_sum)

                    if len(hand_sum) != 0:
                        summation = hand_sum.sum(axis=0) / len(hand_sum)
                        for z in range(3):
                            points_left[json_number][-1][z] = summation[z]

                    # Right Side
                    index = 0
                    for p in [0, 1, 2, 3, 4]:
                        for z in range(3):
                            points_right[json_number][index][z] = data["people"][0]["pose_keypoints_2d"][p * 3 + z]
                        index = index + 1

                    hand_sum = []
                    for p in range(21):
                        hand_position = [0, 0, 0]
                        for z in range(3):
                            hand_position[z] = data["people"][0]["hand_right_keypoints_2d"][p * 3 + z]
                        if 0.0 in hand_position:
                            continue
                        else:
                            hand_sum.append(hand_position)
                    hand_sum = np.array(hand_sum)

                    if len(hand_sum) != 0:
                        summation = hand_sum.sum(axis=0) / len(hand_sum)
                        for z in range(3):
                            points_right[json_number][-1][z] = summation[z]

                json_number = json_number + 1

            # Remove frames containing zero
            c = 0
            while c < len(points_left):
                if 0.0 in points_left[c]:
                    # print(points_left[c], c, "removed by left")
                    points_left = np.delete(points_left, c, axis=0)
                    points_right = np.delete(points_right, c, axis=0)
                else:
                    c = c + 1

            c = 0
            while c < len(points_right):
                if 0.0 in points_right[c]:
                    # print(points_right[c], c, "removed by right")
                    points_left = np.delete(points_left, c, axis=0)
                    points_right = np.delete(points_right, c, axis=0)
                else:
                    c = c + 1

            if len(points_left) < 2 or len(points_right) < 2:
                continue

            # Left Side Body Movement
            difference = points_left[:-1] - points_left[1:]
            difference_square = difference ** 2
            difference_square_sum = difference_square[:, :, 0] + difference_square[:, :, 1]
            distance = difference_square_sum ** 0.5

            move_joint = np.sum(distance, axis=0)
            move_body_left = 0
            for s in range(1, len(move_joint)):
                move_body_left = move_body_left + move_joint[s] * (6 - s)

            move_body_left_average = move_body_left / len(difference)

            # Right Side Body Movement
            difference = points_right[:-1] - points_right[1:]
            difference_square = difference ** 2
            difference_square_sum = difference_square[:, :, 0] + difference_square[:, :, 1]
            distance = difference_square_sum ** 0.5

            move_joint = np.sum(distance, axis=0)
            move_body_right = 0
            for s in range(1, len(move_joint)):
                move_body_right = move_body_right + move_joint[s] * (6 - s)

            move_body_right_average = move_body_right / len(difference)

            # Left Side Angle Change
            angles_left = np.zeros([len(points_left), 4])
            place = 0
            for a in points_left:
                for d in range(1, len(a) - 1):
                    d1 = a[d + 1] - a[d]
                    d2 = a[d - 1] - a[d]

                    d1[1] = -d1[1]
                    d2[1] = -d2[1]
                    d1_angle = np.arctan2(d1[1], d1[0]) * 180 / np.pi
                    d2_angle = np.arctan2(d2[1], d2[0]) * 180 / np.pi
                    angle = abs(d1_angle - d2_angle)

                    angles_left[place][d - 1] = angle

                place = place + 1

            difference_angle = angles_left[:-1] - angles_left[1:]

            angle_joint = np.sum(abs(difference_angle), axis=0)
            angle_body_left = 0
            for s in range(len(angle_joint)):
                angle_body_left = angle_body_left + angle_joint[s] * (4 - s)

            angle_body_left_average = angle_body_left / len(difference_angle)

            # Right Side Angle Change
            angles_right = np.zeros([len(points_right), 4])
            place = 0
            for a in points_right:
                for d in range(1, len(a) - 1):
                    d1 = a[d + 1] - a[d]
                    d2 = a[d - 1] - a[d]

                    d1[1] = -d1[1]
                    d2[1] = -d2[1]
                    d1_angle = np.arctan2(d1[1], d1[0]) * 180 / np.pi
                    d2_angle = np.arctan2(d2[1], d2[0]) * 180 / np.pi
                    angle = abs(d1_angle - d2_angle)

                    angles_right[place][d - 1] = angle

                place = place + 1

            difference_angle = angles_right[:-1] - angles_right[1:]

            angle_joint = np.sum(abs(difference_angle), axis=0)
            angle_body_right = 0
            for s in range(len(angle_joint)):
                angle_body_right = angle_body_right + angle_joint[s] * (4 - s)

            angle_body_right_average = angle_body_right / len(difference_angle)

            # Append results
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Total Body Change Left\"", int(move_body_left)])
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Total Body Change Right\"", int(move_body_right)])
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Average Body Change Left\"", int(move_body_left_average)])
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Average Body Change Right\"", int(move_body_right_average)])

            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Total Angle Change Left\"", int(angle_body_left)])
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Total Angle Change Right\"", int(angle_body_right)])
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Average Angle Change Left\"", int(angle_body_left_average)])
            result_list.append([k[25:], tt[0][0], tt[0][2], tt[0][3], tt[0][4][:-15], "\"Average Angle Change Right\"", int(angle_body_right_average)])

    return result_list


image_folder = "people_images"
results = calculate_body_movement_and_angles(image_folder)

for k in results:
    print(k)

csv_file = open("Results.csv", "w")
csv_file.write("\"Person ID\";Nativeness;Frames;\"String Type\";Manner;Path;\"Value Type\";\"Value\"\n")

for k in results:
    gender = "Not In The Lıst"
    if int(k[0]) in people_native:
        gender = "Native"
    elif int(k[0]) in people_late:
        gender = "Late"
    s = ";"
    string = k[0] + s + gender + s + k[1] + s + k[2] + s + k[3] + s + k[4] + s + k[5] + s + str(k[6]) + "\n"
    csv_file.write(string)

csv_file.close()