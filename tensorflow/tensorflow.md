# tensorflow
tensorflow中采用`tf.image`进行图像处理
```
import tensorflow as tf
img = tf.gfile.FastGFile(image_path,'r').read()  # 读取图片
```
读取得到的`img`为`string`型的`tensor`，需要进行进行解码转换为tensorflow可用的数据类型。

---

### 一.图像编码与解码
一张RGB三通道的彩色图像可以看成一个三维矩阵，矩阵中的数字大小代表图像的像素值。<br>
图像在存储时并不是直接存储这些数字，而是经过了压缩编码。<br>
因此在读取及保存图像时需要解码和编码，类似于opencv中的imread和imwrite。

#### 1.tf.image.decode_jpeg
**函数：** 
```
tf.image.decode_jpeg(img, 
		channels=0, 
		ratio=1, 
		fancy_upscaling=True,
		try_recover_truncated=False,
		acceptable_fraction=1,
		dct_method='',
		name=None)
```
**功能：** 将 JPEG/JPG 图片解码为`uint8`类型的`tensor`。<br>
**参数：** `channels`int型，默认为`0`。0:使用输入图片的通道数；1:输出为灰度图；3:v。<br>
&emsp; &emsp; &thinsp; `ratio`int型，默认为`1`。下采样率。<br>
&emsp; &emsp; &thinsp; `fancy_upscaling`bool型，默认为`True`。若为`True`则会使用更慢但更好的色度平面上采样（仅限于yuv420/yuv422)。<br>
&emsp; &emsp; &thinsp; `try_recover_truncated`bool型，默认为`False`。若为`True`则尝试从一个截断输入中恢复出图片。<br>
&emsp; &emsp; &thinsp; `acceptable_fraction`float型，默认为`1`。在被截断的输入接受之前，所需的最小行数。<br>
&emsp; &emsp; &thinsp; `dct_method`string型，默认为`""`，表示为选择系统默认方法。解压缩方法，可选参数有："INTEGER_FAST"， "INTEGER_ACCURATE"。<br>
&emsp; &emsp; &thinsp; `name`操作名（可选）。

#### 2.tf.image.decode_png
**函数：** 
```
tf.image.decode_png(img,
		channels=0,
		dtype=tf.dtypes.uint8,
		name=None)
```
**功能：** 将 PNG 图片解码为`uint8`类型或`uint16`类型的`tensor`。<br>
**参数：** `channels`int型，默认为`0`。0:使用输入图片的通道数；1:输出为灰度图；3:输出为RGB图。<br>
&emsp; &emsp; &thinsp; `dtype`tf.DType型，默认为`tf.uint8`。可选参数有：tf.uint8，tf.uint16。<br>
&emsp; &emsp; &thinsp; `name`操作名（可选）。

#### 3.tf.image.decode_bmp
**函数：** 
```
tf.image.decode_bmp(img,
		channels=0,
		name=None)
```
**功能：** 将 PNG 图片解码为`uint8`类型的`tensor`。<br>
**参数：** `channels`int型，默认为`0`。0:使用输入图片的通道数；1:输出为灰度图；3:输出为RGB图。<br>
&emsp; &emsp; &thinsp; `name`操作名（可选）。

#### 4.tf.image.decode_gif
**函数：** 
```
tf.image.decode_gif(img,
		name=None)
```
**功能：** 将 GIF 图像的第一帧解码为`uint8`类型的`tensor`。<br>

#### 5.tf.image.decode_image
**函数：** 
```
tf.image.decode_image(img
		channels=None,
		dtype=tf.dtypes.uint8,
		name=None)
```
**功能：** 将 BMP/GIF/JPEG/JPG/PNG 图片采用合适操作转换为`dtype`型的`tensor`
