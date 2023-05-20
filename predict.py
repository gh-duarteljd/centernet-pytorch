#------------------------------------------------- ----------------------#
# predict.py will single picture prediction, camera detection, FPS test and directory traversal detection and other functions
# Integrate into a py file, and modify the mode by specifying the mode.
#------------------------------------------------- ----------------------#
import time

import cv2
import numpy as np
from PIL import Image

from centernet import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    #------------------------------------------------- -------------------------------------------------- -------#
    # mode is used to specify the mode of the test:
    # 'predict' means single image prediction, if you want to modify the prediction process, such as saving images, intercepting objects, etc., you can read the detailed notes below
    # 'video' means video detection, you can call the camera or video for detection, see the note below for details.
    # 'fps' means test fps, the image used is street.jpg in img, see the note below for details.
    # 'dir_predict' means to traverse the folder to detect and save. By default, the img folder is traversed and the img_out folder is saved. See the note below for details.
    # 'heatmap' represents the heat map visualization of the prediction results, see the notes below for details.
    # 'export_onnx' means to export the model as onnx, which requires pytorch1.7.1 or above.
    #------------------------------------------------- -------------------------------------------------- -------#
    mode = "predict"
    #------------------------------------------------- ------------------------#
    # crop specifies whether to intercept the target after a single image prediction
    # count specifies whether to count the target
    # crop and count are only valid when mode='predict'
    #------------------------------------------------- ------------------------#
    crop = False
    count = False
    #------------------------------------------------- -------------------------------------------------- -------#
    # video_path is used to specify the path of the video, when video_path=0, it means to detect the camera
    # If you want to detect video, you can set it like video_path = "xxx.mp4", which means to read the xxx.mp4 file in the root directory.
    # video_save_path indicates the path to save the video, when video_save_path="" means not to save
    # If you want to save the video, you can set it like video_save_path = "yyy.mp4", which means saving it as a yyy.mp4 file in the root directory.
    # video_fps fps for saved video
    #
    # video_path, video_save_path and video_fps are only valid when mode='video'
    # When saving the video, you need to ctrl+c to exit or run until the last frame to complete the complete saving step.
    #------------------------------------------------- -------------------------------------------------- -------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    #------------------------------------------------- -------------------------------------------------- -------#
    # test_interval is used to specify the number of image detections when measuring fps. In theory, the larger the test_interval, the more accurate the fps.
    # fps_image_path is used to specify the fps image of the test
    #
    # test_interval and fps_image_path are only valid in mode='fps'
    #------------------------------------------------- -------------------------------------------------- -------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    #------------------------------------------------- ------------------------#
    # dir_origin_path specifies the folder path of the image used for detection
    # dir_save_path specifies the save path of the detected image
    #
    # dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    #------------------------------------------------- ------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    #------------------------------------------------- ------------------------#
    # heatmap_save_path The save path of the heat map, which is saved under model_data by default
    #
    # heatmap_save_path is only valid in mode='heatmap'
    #------------------------------------------------- ------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #------------------------------------------------- ------------------------#
    # simplify use Simplify onnx
    # onnx_save_path specifies the save path of onnx
    #------------------------------------------------- ------------------------#
    simplify=True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        '''
        1. If you want to save the detected image, use r_image.save("img.jpg") to save it, and modify it directly in predict.py.
        2. If you want to get the coordinates of the prediction frame, you can enter the centernet.detect_image function and read the four values of top, left, bottom, and right in the drawing part.
        3. If you want to use the prediction frame to intercept the target, you can enter the centernet.detect_image function, and use the obtained four values of top, left, bottom, and right in the drawing part
        Use the matrix method to intercept on the original image.
        4. If you want to write additional words on the prediction map, such as the number of specific targets detected, you can enter the centernet.detect_image function and judge the predicted_class in the drawing part.
        For example, judge if predicted_class == 'car': to judge whether the current target is a car, and then record the quantity. Use draw.text to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image. open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = centernet. detect_image(image, crop = crop, count=count)
                r_image. show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture. read()
        if not ref:
            raise ValueError("Failed to read the camera (video) correctly, please pay attention to whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        while(True):
            t1 = time. time()
            # read a frame
            ref, frame = capture. read()
            if not ref:
                break
            # Format conversion, BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # check
            frame = np.array(centernet.detect_image(frame))
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
            fps = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out. write(frame)

            if c==27:
                capture. release()
                break

        print("Video Detection Done!")
        capture. release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out. release()
        cv2.destroyAllWindows()
    
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + 'seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = centernet. detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image. open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                centernet. detect_heatmap(image, heatmap_save_path)
    
    elif mode == "export_onnx":
        centernet.convert_to_onnx(simplify, onnx_save_path)
    
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")