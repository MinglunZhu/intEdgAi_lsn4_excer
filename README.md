# intEdgAi_lsn4_excer
## Description
This is code I wrote for Udacity Course Intel Edge AI Lesson 4 Excercise - Integrate into An App.

It's a Python commandline program that uses Intel OpenVINO Toolkit for vehicle detection. It detects the location of vehicles in the images of the video and then draws bounding boxes on each detected vehicle in each image, puts the images back together, then outputs the video with bounding boxes.

***Please note: This is not a working demo, it is example code mainly for you to read through.
You will need OpenVINO installed, and you will also need to change the hardcoded location of the OpenVINO file to run the program.***

### Model
The model I used is the pre-trained model `vehicle-detection-adas-binary-0001` from the OpenVINO model zoo, which is already converted to the OpenVINO intermediate representation.

This model is not very good as you can see from its outputs - you have to set confidence threshold to really high in order to reduce misclassifications, but in return you also have fewer detected vehicles, so not all vehicles are detected.

### Parameters
The program allows you to choose the input video, the confidence threshhold for detected vehicles, and the color of the bounding box. Use `-h` to see the parameter info.

## Example Output
The program takes in a video, by default, the provided test video, and outputs video with bounding boxes drawn with confidence threshold of `.5`. An example output video is provided in the `Example Output` folder which used confidence threshold of `.9`.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
