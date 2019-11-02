import sys
import os

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("--------------------------------")
        print("This script receives a directory that contains a file 'bounding_boxes.txt' and rewrites them in the format expected by the WIDER matlab eval script")
        print("")
        print("Usage: ")
        print("python output_to_wider_eval_format.py 'output_folder' ")
        print("--------------------------------")

    else:

        # Read bounding boxes from test file
        bb_filename_test = sys.argv[1] + "/bounding_boxes.txt"
        bb_file_test = open(bb_filename_test, "r")
        bboxes_test = {}

        while (True):
            image_name = bb_file_test.readline().strip()
            if not image_name:
                break
            bboxes_test[image_name] = []

            num_bboxes = int(bb_file_test.readline().strip())
            for j in range(num_bboxes):
                bbox = bb_file_test.readline().strip().split(",")
                #bbox[2] = int(bbox[0]) + int(bbox[2])
                #bbox[3] = int(bbox[1]) + int(bbox[3])
                bboxes_test[image_name].append(bbox)
                
        eval_output_dir = sys.argv[1] + "/eval_output"
        if not os.path.exists(eval_output_dir):
            os.mkdir(eval_output_dir)

        for image_name in bboxes_test:
            file_name = os.path.splitext(image_name)[0] + ".txt"
            output_file = open(eval_output_dir + "/" + file_name, "w")
            output_file.write(os.path.splitext(image_name)[0] + "\n")
            output_file.write(str(len(bboxes_test[image_name])) + "\n")
            for i in range(len(bboxes_test[image_name])):
                bbox = bboxes_test[image_name][i]
                output_file.write(" ".join(bbox) + " 1.000" + "\n")
            output_file.write("\n")
            output_file.close()