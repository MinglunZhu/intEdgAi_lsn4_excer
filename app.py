import argparse
import cv2
from inference import Network

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
COLORS = {
    "blue": (255,0,0), 
    "green": (0,255,0), 
    "red": (0,0,255)
}

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    conf_desc = 'The confidence threshold'
    col_desc = 'The color for bounding boxes: red, green, or blue'

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-conf", help = conf_desc, default = .5)
    optional.add_argument("-col", help = col_desc, default = 'blue')
    
    args = parser.parse_args()

    return args

def convert_color(col):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    color = COLORS.get(col)
    
    if color:
        return color
    else:
        return COLORS['BLUE']
    
def draw_boxes(frame, boxes, width, height, conf, col):
    '''
    Draw bounding boxes onto the frame.
    '''
    # Output shape is n x 7
    #[image_id, label, conf, x_min, y_min, x_max, y_max]
    for box in boxes:
        if box[2] >= conf:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), col, 1)
            
    return frame

def preProc(img, w, h):
    i = cv2.resize(img, (w, h))
    
    #change dim to c, h, w
    i = i.transpose((2, 0, 1))
    
    #add batch dim
    return [i]

def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    ntw = Network()

    ### TODO: Load the network model into the IE
    ntw.load_model(args.m, args.d, CPU_EXTENSION)
    iptShape = ntw.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    #reset frame count
    frameCnt = 0
    col = convert_color(args.col)
    conf = float(args.conf)
    frames = {}
    
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        
        if not flag:
            #wait for all inference to complete
            for i in range(frameCnt):
                rqs = ntw.exec_network.requests[i]
                status = rqs.wait()
                
                ### TODO: Get the output of inference
                #if inference was successful, draw boxes
                #if not successful, do nothing, no box drawn, and just output original frame
                f = frames[i]
                
                if status == 0:
                    boxes = rqs.outputs['detection_out'][0][0]

                    ### TODO: Update the frame to include detected bounding boxes
                    f = draw_boxes(f, boxes, width, height, conf, col)

                # Write out the frame
                out.write(f)
            
            # Release the out writer and destroy any OpenCV windows
            out.release()
            cv2.destroyAllWindows()
                
            print('total frames: {}'.format(frameCnt))
            
            break
            
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        frames[frameCnt] = frame
        
        ppImg = preProc(frame, iptShape[3], iptShape[2])

        ### TODO: Perform inference on the frame
        rqs = ntw.async_inference(frameCnt, ppImg)

        #there is no point in performing an async request,
        #if we wait for the inference to complete before processing the next frame
        #intead, moved all waiting to the end of the while loop
        frameCnt += 1
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture
    cap.release()

def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
