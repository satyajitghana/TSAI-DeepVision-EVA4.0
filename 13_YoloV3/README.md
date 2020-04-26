
# YoloV3

## Assignment
Assignment:

1.  OpenCV Yolo:  [SOURCE (Links to an external site.)](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
    1.  Run this above code on your laptop or Colab.
    2.  Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn).
    3.  Run this image through the code above.
    4.  Upload the link to GitHub implementation of this
    5.  Upload the annotated image by YOLO.
2.  Training Custom Dataset on Colab for YoloV3
    1.  Refer to this Colab File:  [LINK (Links to an external site.)](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
    2.  Refer to this GitHub  [Repo (Links to an external site.)](https://github.com/theschoolofai/YoloV3)
    3.  Collect a dataset of 500 images and annotate them.  **Please select a class for which you can find a YouTube video as well.** Steps are explained in the readme.md file on GitHub.
    4.  Once done:
        1.  [Download (Links to an external site.)](https://www.y2mate.com/en19)  a very small (~10-30sec) video from youtube which shows your class.
        2.  Use  [ffmpeg (Links to an external site.)](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)  to extract frames from the video.
        3.  Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
        4.  Inter on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub.  
            `python detect.py --conf-thres 0.3 --output output_folder_name`
        5.  Use  [ffmpeg (Links to an external site.)](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)  to convert the files in your output folder to video
        6.  Upload the video to YouTube.
    5.  Share the link to your GitHub project with the steps as mentioned above
    6.  Share the link of your YouTube video


# YOLO V3 on OpenCV

Colab Notebook: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/YoloV3-OpenCV.ipynb
Github Link: https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/YoloV3-OpenCV.ipynb

### Doggo
![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/opencv-run/doggo.png?raw=true)

### Me
![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/opencv-run/me.png?raw=true)

### Potto
![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/opencv-run/potto.png?raw=true)

# Training YOLO V3 on a Custom Dataset

Github Link: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/YoloV3-BugsBunny.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/YoloV3-BugsBunny.ipynb)
Colab Notebook: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/13_YoloV3/YoloV3-BugsBunny.ipynb

# Running the custom model on a video

[![BUGS BUNNY](https://img.youtube.com/vi/R8sgaO6AJyE/0.jpg)](https://www.youtube.com/watch?v=R8sgaO6AJyE)
