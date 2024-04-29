# Image dimensions


![img.png](img.png)

Keep in mind when reading images with opencv, images are read in matrix notation.
The dimensions of a matrix are (rows, columns) - which is (height, width) in image notation.<br />

Therefore, the following expression applies:

_&theta;<sub>mat</sub><exp>T</exp>  = &theta;<sub>img</sub> 	&and; &theta;<sub>img</sub><exp>T</exp>  = &theta;<sub>mat</sub>_<br />
_where &theta;_ &ensp; &ensp;    is an Image <br />
&ensp; &ensp; &ensp; &ensp; _img_ &ensp;represents the image notation <br />
&ensp; &ensp; &ensp; &ensp; _mat_ &ensp;represents the matrix notation <br />

To ensure consistency throughout any image operation - **Images are always represented in image notation**.
The image reader will always return _&theta;<sub>mat</sub><exp>T</exp>_. This should also solve the problem with opencv, 
where images are read as bgr instead of rgb.

When returning augmented images, the rgb-matrix notation is applied. Here only the first and the second image dimensions 
are transposed.

