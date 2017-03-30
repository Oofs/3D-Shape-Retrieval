# 3D-Shape-Retrieval

This is one project of the 3rd year engineer student at Ecole Centrale de Lyon.
The goal of our project is to achieve 3D object retrieval by applying CNN on a dataset of 3D objects. The idea is not using directly the 3D mesh itself but using the projected-based apparences: like RGB images or depth images of these 3D meshes. This approche is proposed and tested by several teams on a competition SHREC2016 : http://shapenet.cs.stanford.edu/shrec16/ In this link, you will see more details about the CNN + Multi-view based approche.
In our project, we build our own CNN with Tensorflow and test it on the data of SHAPENET CORE55.

To run our code, you should :

1. Install the Tensorflow on your computer. With this tutorial : https://www.tensorflow.org/install/ it shouldn't be difficult for you

2. Download the database of 2D images with multi-view with the link in /data/downloadlink

3. Transfer the 2D images to the format MINIST (see our tutorial in /data)

4. Read our tutorial and run our code on the Tensorflow.


The structure of CNN which we used is : [[conv - relu]x3 - pool]x3 - fullyconnected - relu - fullyconnected - softmax.
By now, we did our test on a little database and we achieved a precision which is not very impressive : 42.7%. Next, we should launch our test  on a larger database and try more CNN structures with a more powerful server. 



