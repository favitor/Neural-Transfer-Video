import argparse
import time
import os
import subprocess
import cv2 as cv

FLAGS = None
VID = 'video'
path = './styleimages'

def predict(img, h, w):
    blob = cv.dnn.blobFromImage(img, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)

    print ('[INFO] Setting the input to the model')
    net.setInput(blob)

    print ('[INFO] Starting Inference!')
    start = time.time()
    out = net.forward()
    end = time.time()
    print ('[INFO] Inference Completed successfully!')

    # Reshape the output tensor and add back in the mean subtraction, and
    # then swap the channel ordering
    out = out.reshape((3, out.shape[2], out.shape[3]))
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.680
    out /= 255.0
    out = out.transpose(1, 2, 0)

    # Printing the inference time
    if FLAGS.print_inference_time:
        print ('[INFO] The model ran in {:.4f} seconds'.format(end-start))

    return out

# Source for this function:
# https://github.com/jrosebr1/imutils/blob/4635e73e75965c6fef09347bead510f81142cf2e/imutils/convenience.py#L65
def resize_img(img, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    #I've some troubles to always get the image size
    try:
        h, w = img.shape[:2]
        if h or w is None:
            h = img.shape[0]
            w = img.shape[1]
        else:
            h = int(vid.get(3))
            w = int(vid.get(4))

    except:
        height, width, _ = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(img, dim, interpolation=inter)
    return resized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path',
                type=str,
                default='./models/instance_norm/',
                help='The model directory.')

    parser.add_argument('-i', '--video',
                type=str,
                help='Path to the video.')

    parser.add_argument('-md', '--model',
                type=str,
                help='The file path to the direct model.\
                 If this is specified, the model-path argument is \
                 not considered.')

    parser.add_argument('--download-models',
                type=bool,
                default=False,
                help='If set to true all the pretrained models are downloaded, \
                    using the script in the downloads directory.')

    parser.add_argument('--print-inference-time',
                type=bool,
                default=False,
                help='If set to True, then the time taken for the model is output \
                    to the console.')

    FLAGS, unparsed = parser.parse_known_args()

    # download models if needed
    if FLAGS.download_models:
        subprocess.call(['./models/download.sh'])

    # Set the mode image/video based on the argparse
    if FLAGS.video is None:
        print("Error, please try again")
    else:
        mode = VID
    # Check if there are models to be loaded and list them
    models = []
    for f in sorted(os.listdir(FLAGS.model_path)):
        if f.endswith('.t7'):
            models.append(f)

    if len(models) == 0:
        raise Exception('The model path doesn\'t contain models')

    # Load the neural style transfer model
    path = FLAGS.model_path + ('' if FLAGS.model_path.endswith('/') else '/')
    print (path + models[0])
    print ('[INFO] Loading the model...')

    model_loaded_i = -1
    total_models = len(os.listdir(FLAGS.model_path))

    if FLAGS.model is not None:
        model_to_load = FLAGS.model
    else:
        model_loaded_i = 0
        model_to_load = path + models[model_loaded_i]
    net = cv.dnn.readNetFromTorch(model_to_load)

    print ('[INFO] Model Loaded successfully!')

    # Loading the image depending on the type
    if mode == VID:
        pass
        vid = cv.VideoCapture(FLAGS.video)
        #Or you can use
        #vid = cv.VideoCapture("yourvideo.mp4")
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))
        count = 0
        while True:
            _, frame = vid.read()
            img = resize_img(frame, width=600)
            h, w  = img.shape[:2]
            h = frame_height
            w = frame_width
            out = predict(img, h, w)
            cv.imwrite(os.path.join(path, "frame%d.jpg" % count), out)
            cv.imwrite("text.jpg", img)
            cv.imwrite("test.jpg", out)
            
            #cv.imshow('Stylizing Real-time Video', out)
            #count += 1

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and FLAGS.model is None:
                model_loaded_i = (model_loaded_i + 1) % total_models
                model_to_load = path + models[model_loaded_i]
                net = cv.dnn.readNetFromTorch(model_to_load)
            elif key == ord('p') and FLAGS.model is None:
                model_loaded_i = (model_loaded_i - 1) % total_models
                model_to_load = path + models[model_loaded_i]
                net = cv.dnn.readNetFromTorch(model_to_load)

        #fourcc = cv.VideoWriter_fourcc(*'XVID')
        #finalvideo = cv.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))
        vid.release()
        cv.destroyAllWindows()

