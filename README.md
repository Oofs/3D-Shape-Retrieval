# 3D-Shape-Retrieval

This is one project of the 3rd year engineer student at Ecole Centrale de Lyon.
The goal of our project is to achieve 3D object retrieval by applying CNN on a dataset of 3D objects. The idea is not using directly the 3D mesh itself but using the projected-based apparences: like RGB image or depth images of these 3D meshes. This approche is proposed and tested by several teams on a competition SHREC2016:http://shapenet.cs.stanford.edu/shrec16/ In this link, you will see more details about the CNN + Multi-view based approche.
In our project, we build our own CNN with Tensorflow and test it on the data of SHAPENET CORE55.

To run our code, you should :

1. Install the Tensorflow on your computer. With this tutorial :https://www.tensorflow.org/install/ it shouldn't be difficult for you

2. Download the database of 2D images with multi-view with the link in /data/downlink

3. Run our code on the Tensorflow and have fun !
