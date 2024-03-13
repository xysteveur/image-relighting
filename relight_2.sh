#!/bin/bash
for filename in data/test/train_bg_blc/*.jpg; do
    python relight_2.py --source_image $(basename $filename) --light_image light_2.png --model trained.pt --gpu --face_detect both --crop_paste
done

# python test_network.py --image 0ac176bc-keys-4-02.jpg --model trained.pt --gpu


# for filename in data/test/train_bg_blc/*.jpg; do
#     # for ((i=0; i<=3; i++)); do
#     #     ./MyProgram.exe "$filename" "Logs/$(basename "$filename" .txt)_Log$i.txt"
#     # done
#     python test_network.py --image $(basename $filename) --model trained.pt --gpu
# done


