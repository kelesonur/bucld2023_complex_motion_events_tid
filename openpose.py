import os
import subprocess


def run_openpose(image_folder):
    info = os.listdir(image_folder)

    info.sort()

    print(info)

    for k in info:

        frames_folder = image_folder + "/" + k

        if os.path.isdir(frames_folder + "/json"):
            continue

        command = ["bin/OpenPoseDemo.exe", "--image_dir", frames_folder, "--write_json",
                   frames_folder + "/json", "--hand", "--net_resolution", "-1x256"]

        print(command)

        process_output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(process_output.stderr) != 0:
            print("Error in command: ", command)
        answer = process_output.stdout.decode('utf8', errors='strict')
        print(answer)


image_folder = "people_images"

run_openpose(image_folder)
