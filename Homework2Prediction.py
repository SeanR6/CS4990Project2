#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

test_model = load_model('ModelData/GenderModel.h5')

test_data_dir = 'Project2Datasets/Gender/test'

with open('output.csv', 'w') as file:
    writer=csv.writer(file, delimiter=',', lineterminator='\n',)
    row = ["Id"] + ["Expected"]
    for filename in os.listdir(test_data_dir):
        print(filename)
        img = image.load_img((test_data_dir+'/'+filename),False, target_size=(150,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = x
        classes = test_model.predict(images)
        print(classes[0,0])
        row = [filename] + [classes[0,0]]
        writer.writerow(row)
    


# In[ ]:




