#!/bin/bash
VIDEO_DIR="/home/thuratun/GW_workspace/CS2/Hitachi_Astemo_Prj/MONO_Lss/MonoLSS/kitti/inference_data/selected_video_for_infer/"
WEIGHT_FILE="./pretrained_model/50_tensor(0.4534)_lane_detection_network.pth"
DEVICE="cuda:0"
SAVE_PATH="./infer_output"

# Check if the directory exists
if [ ! -d "${VIDEO_DIR}" ]; then
    echo "Directory '${VIDEO_DIR}' does not exist."
    exit 1
fi

# Initialize an empty array to store filenames
file_list=()

# Load files into the array
while IFS= read -r -d '' file; do
    file_list+=("$file")
done < <(find "${VIDEO_DIR}" -maxdepth 1 -type f -print0)

# Check if no files were found
if [ ${#file_list[@]} -eq 0 ]; then
    echo "No files found in '${VIDEO_DIR}'."
    exit 1
fi

for file in "${file_list[@]}"; do
    python3 tools/run_infer.py --input-video ${file} --weight-file $WEIGHT_FILE --device $DEVICE --save-result
done