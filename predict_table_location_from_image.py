from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.predicting import PredictorNetwork
from PIL import Image as pilimage

# This program  will predict the location of the tables in an image
# It outputs the coordinates of the tables. Using these coordinates we can cut the table portion of the image and use it for further processing

input_file = '/usr/local/table-detection-from-images-using-deep-learning-master/data/val/9549_009.png'
# Specify the luminoth checkpoint here
checkpoint = '73689cee5da2'

config = get_checkpoint_config(checkpoint)
network = PredictorNetwork(config)
image = pilimage.open(input_file).convert('RGB')
objects = network.predict_image(image)

print("NO OF TABLES IDENTIFIED BY LUMINOTH = " + str(len(objects)))
print('-' * 100)

table_counter = 1

for i in range(len(objects)):
    table_idctionary = objects[i]
    coordinate_list = table_idctionary["bbox"]
    xminn = coordinate_list[0]
    yminn = coordinate_list[1]
    xmaxx = coordinate_list[2]
    ymaxx = coordinate_list[3]
    print('TABLE ' + str(table_counter) + ':')
    print('-' * 100)
    print("xminn = " + str(xminn))
    print("yminn = " + str(yminn))
    print("xmaxx = " + str(xmaxx))
    print("ymaxx = " + str(ymaxx))
    table_counter += 1
