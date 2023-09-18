### For 3D Reconstruction
> all.py -> 3D_reconstruction.ipynb -> err.py

all.py
 - contain all the process including preprocessing, cnn prediction to find the corresponding pair
   
camera.py
 - capture webcam picture
 
cut.py
 - cut each pattern from the picture
  
cnn_predict.py
 - predict the pattern number
 
direction.py
 - calculate the direction of each pattern
 
match.py
 - find the nieghbors of each pattern and match with the hash table

3D_Reconstruction.ipynb
  - reconstruct the point cloud with the corresponding pair

draw.py
 - DrawCenter : draw the center of the pattern that are decoded
 - DrawDirection : draw the direction of each pattern
 - DrawFinal : draw the final pattern that are matched
   
preprocess.py
 - pre: preprocess the image with CLAHE and Otsu's method
 - split: preprocess the image with spliting the image into nine zones

### For error calculating

ransac_circle.py
 - fit a circle with randan sample

err.py
 - calculate the error of fitting a plane and fitting a circle
   
### For pattern creation
> encode.py -> pattern_8.py
encode.py
 - create the DB sequence encoding
 - decode function also inside this file

pattern_8.py
 - create pattern with specfic size
   
### For training data
> pattern_8.py -> data_collection.py -> stereo_projection.py -> cnn.py
data_collection.py
 - collect the data to train the cnn model

stereo_projecction.py
 - stereo projection

cnn.py
 - train for cnn model
