# Computer-vision
Digital attendance system based on group picture using PCA algorithm and HaarCascade Algorithm 


The codes submitted corresponds to creating a face recognition system based on PCA algorithm 
and using basic Linear Algebra. 

The code consists of two parts 
1) The face recognition part which will recognise faces based on PCA analysis
2) The group.py which will extract faces from the group first, so that I can apply the face recognition algorithm on the faces.

Overall this project aims at dimension reduction and face recognition using basic PCA and Linear Algebra. 


## Task:

Digital attendance system based on group picture

Suppose an instructor takes a group picture of students sitting in each class for at-
tendance purpose. It is hard to manually check the faces for a a large class. Code the
possible solution for this problem using PCA, SVD, LDA or t-SNE

## Approach:

We take the help of Linear Algebra to help us solve this problem (Principle Component Analysis
(PCA) to be specific).

###  Getting the Data:

We downloaded our face-dataset from https://cam-orl.co.uk/facedatabase.html (formerly ’The
ORL Database of Faces’). There are ten different images of each of 40 distinct subjects. For
some subjects, the images were taken at different times, varying the lighting, facial expressions (open
/ closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were
taken against a dark homogeneous background with the subjects in an upright, frontal position (with
tolerance for some side movement).

### Pre-processing:

We didn’t had to do any pre-processing of the image data downloaded as it was already in the usable
formatPGM(PGM file is a grayscale image file saved in the portable gray map (PGM) format and
encoded with one or two bytes (8 or 16 bits) per pixel) with proper cropping and hence we imported
our whole data-set without any pre-processing.


### Analysis on the image data:

####  Vectorisation of Image:

We start by Vectorising each image into a vector so that it’s easier to work with them and apply all
the numerical data analysis tools.

```
faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.
IMREAD_GRAYSCALE)
```

We extract each PGM file into a byte string through image.read() and convert it into anumpy
array of bytes. Then we use OpenCV to decode the byte string into an array of pixels using
cv2.imdecode(). The file format will be detected automatically by OpenCV. We save each picture
into a Python dictionary faces for later use.

####  Subtracting the means:

The mean subtracted is the average across each dimension. So, all thexvalues have ̄x(the mean of
thexvalues of all the data points) subtracted, and all theyvalues have ̄ysubtracted from them.This
produces a data set whose mean is zero.

####  Calculate the Covariance matrix:


We know that

![image](https://user-images.githubusercontent.com/66754219/221323600-f99923cd-d9d4-4bdf-ab57-722c4650064e.png)


We calculate the covariance matrix to get to know about how much is the overlap of all the images
with one other so that we can eliminate these redundancy.

####  Calculate the eigenvectors and eigenvalues of the covariance matrix

We now calculate the eigenvectors which are orthogonal to each other and hence I have gotten a
basis set where all the axes are orthogonal to each other and there is no dependency on each other
(i.e. Linearly Independent). Now I can plot the variances which is captured by all the eigenvectors
and I see that only a small proportion of eigenvectors hold a significant amount of variance and all
others are pretty insignificant.



![image](https://user-images.githubusercontent.com/66754219/221322238-295b9d99-38c9-47e7-8915-40659e556102.png)


####  Deriving the new data set

This the final step in PCA, and is also the easiest. Once we have chosen the components (eigenvectors)
that we wish to keep in our data and formed a feature vector, we simply take the transpose of the
vector and multiply it on the left of the original data set, transposed.
What will this give us? It will give us the original data solely in terms of the vectors we chose.
If these axes are perpendicular, then the expression is the most efficient. This was why it was
important that eigenvectors are always perpendicular to each other.


![image](https://user-images.githubusercontent.com/66754219/221322433-f17e790f-7d66-47b3-b349-68f05d9bac8b.png)


As we can see that the re-constructed faces are not 100% similar because we have used only a
certain fraction of information from them to re-construct the data, but we can clearly recognize the
faces which is a great achievement and proves the fact that we can work with the limited set of
information which we chose.

###  Providing Image for query and further analysis:

Now, we put a related image into our code and see whether it can match to it’s corresponding face
or not.
For matching we use theEuclidean Distance Algorithmto choose the closest match.


![image](https://user-images.githubusercontent.com/66754219/221322549-3f2dab2f-3690-48de-ba43-e5fb31417094.png)


Here, we see that although this method works alright in most of the cases, but this method of
euclidean distance will at times give an incorrect match as can be seen in the image above. Hence,
we come up with aTHRESHOLDwithin which we will call an image to be a match and outside
which we will call ”No matching image found in the data-set”.

###  Getting individual images from a group photo:

Here, we use python library OpenCV to detect and extract individuals from a group photo. It
matches the faces with the help ofHaarCascade Algorithm.
It is an Object Detection Algorithm used to identify faces in an image or a real time video. The
algorithm uses edge or line detection features proposed by Viola and Jones in their research paper
“Rapid Object Detection using a Boosted Cascade of Simple Features” published in 2001.


![image](https://user-images.githubusercontent.com/66754219/221322625-10967f53-af7f-4fa3-9889-12eaf1e451f1.png)



Haar Cascade Detection is one of the oldest yet powerful face detection algorithms invented. It has
been there since long, long before Deep Learning became famous. Haar Features were not only used
to detect faces, but also for eyes, lips, license number plates etc. The models are stored on GitHub,
and we can access them with OpenCV methods. But this method is not 100% effective and there might be chances that you might miss some faces in a group, and Hence, it is necessary that for this
Computer Vision attendancesystem to work, the images which are provided into the system
should not have a lot variables to work upon, or else it might not work properly (unless we have an
error proof method of extracting each and every face).

##  Results:

We were able to extract individual photo from a group photo and then pass that onto
the face-matching algorithm and match the face.

##  Conclusion:

We would like to conclude that PCA is a very powerful and reliant method for face recognition and
detection and it can be applied into a Computer Vision based Digital attendance system.

We would also like to conclude that this system needs a lot of work to be done, particularly in
the area of extracting images from group photo

##  References:

- https://cam-orl.co.uk/facedatabase.htmlFace Dataset
- https://github.com/shantnu/FaceDetect/Face Capturing from Group Photo
- [http://staff.ustc.edu.cn/zwp/teach/MVA/pcaface.pdfFace](http://staff.ustc.edu.cn/zwp/teach/MVA/pcaface.pdfFace) Matching Algorithm Guide
- [http://www.cs.otago.ac.nz/cosc453/student-tutorials/principal-components.pdf](http://www.cs.otago.ac.nz/cosc453/student-tutorials/principal-components.pdf)



