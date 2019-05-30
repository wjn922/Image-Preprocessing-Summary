# tensorflow
tensorflow中采用`tf.image`进行图像处理
```
import tensorflow as tf
contents = tf.gfile.FastGFile(image_path,'r').read()  # 读取图片
```

读取得到的`contents`为`string`型的 0-D `tensor`，需要进行解码转换为tensorflow可用的数据类型。

------

## 一.图像编码与解码
一张RGB三通道的彩色图像可以看成一个三维矩阵，矩阵中的数字大小代表图像的像素值。<br>
图像在存储时并不是直接存储这些数字，而是经过了压缩编码。<br>
因此在读取及保存图像时需要解码和编码，类似于opencv中的imread和imwrite。

### 1. tf.image.decode_jpeg
#### 函数： 
```
tf.image.decode_jpeg(contents, 
		channels=0, 
		ratio=1, 
		fancy_upscaling=True,
		try_recover_truncated=False,
		acceptable_fraction=1,
		dct_method='',
		name=None)
```
#### 功能：
将 JPEG/JPG 图片解码为`uint8`类型的`tensor`。<br>
#### 参数： 
`channels` int型，默认为`0`。0:使用输入图片的通道数； 1:输出为灰度图； 3:输出为RGB图。<br>
`ratio` int型，默认为`1`。下采样率，可选参数有1,2,4,8。<br>
`fancy_upscaling` bool型，默认为`True`。若为`True`，则会使用更慢但更好的色度平面上采样（仅限于yuv420/yuv422)。<br>
`try_recover_truncated` bool型，默认为`False`。若为`True`，则尝试从一个截断输入中恢复出图片。<br>
`acceptable_fraction` float型，默认为`1`。在被截断的输入接受之前，所需的最小行数。<br>
`dct_method` string型，默认为`""`，表示为选择系统默认方法。解压缩方法，可选参数有："INTEGER_FAST", "INTEGER_ACCURATE"。<br>
`name` 操作名（可选）。

### 2. tf.image.decode_and_crop_jpeg
#### 函数：
```
tf.image.decode_and_crop_jpeg(contents, 
		crop_window,
		channels=0, 
		ratio=1, 
		fancy_upscaling=True,
		try_recover_truncated=False,
		acceptable_fraction=1,
		dct_method='',
		name=None)
```
#### 功能：  
将 JPEG/JPG 图片解码并裁剪为`uint8`类型的`tensor`。<br>
#### 参数：
`crop_window` int32型的一维tensor。图片需要裁剪的区域: [crop_x, crop_y, crop_height, crop_width]。<br>
`channels` int型，默认为`0`。0:使用输入图片的通道数； 1:输出为灰度图； 3:输出为RGB图。<br>
`ratio` int型，默认为`1`。下采样率，可选参数有1,2,4,8。<br>
`fancy_upscaling` bool型，默认为`True`。若为`True`，则会使用更慢但更好的色度平面上采样（仅限于yuv420/yuv422)。<br>
`try_recover_truncated` bool型，默认为`False`。若为`True`，则尝试从一个截断输入中恢复出图片。<br>
`acceptable_fraction` float型，默认为`1`。在被截断的输入接受之前，所需的最小行数。<br>
`dct_method` string型，默认为`""`，表示为选择系统默认方法。解压缩方法，可选参数有："INTEGER_FAST"，"INTEGER_ACCURATE"。<br>
`name` 操作名（可选）。


### 3.　tf.image.decode_png
#### 函数：
```
tf.image.decode_png(contents,
		channels=0,
		dtype=tf.dtypes.uint8,
		name=None)
```
#### 功能：
将 PNG 图片解码为`uint8`类型或`uint16`类型的`tensor`。<br>
#### 参数：
`channels` int型，默认为`0`。0:使用输入图片的通道数；1:输出为灰度图；3:输出为RGB图。<br>
&emsp; &emsp; &thinsp; `dtype` tf.DType型，默认为`tf.uint8`。可选参数有：tf.uint8，tf.uint16。<br>
&emsp; &emsp; &thinsp; `name` 操作名（可选）。

### 4.　tf.image.decode_bmp
#### 函数：
```
tf.image.decode_bmp(contents,
		channels=0,
		name=None)
```
#### 功能：
 将 PNG 图片解码为`uint8`类型的`tensor`。<br>
#### 参数：
`channels` int型，默认为`0`。0:使用输入图片的通道数； 1:输出为灰度图； 3:输出为RGB图。<br>
`name`操作名（可选）。

### 5.　tf.image.decode_gif
#### 函数：
```
tf.image.decode_gif(contents,
		name=None)
```
#### 功能：
将 GIF 图像的第一帧解码为`uint8`类型的`tensor`。<br>
#### 参数：
`name` 操作名（可选）。

### 6.　tf.image.decode_image
#### 函数：
```
tf.image.decode_image(contents,
		channels=None,
		dtype=tf.dtypes.uint8,
		name=None)
```
#### 功能：
将 BMP/GIF/JPEG/JPG/PNG 图片采用合适操作转换为`dtype`型的`tensor`。<br>
*注：解码 GIF 图片返回的是4-D数组 [num_frames, height, width, 3]，解码其他类型图片返回的是3-D数组 [height, width, num_channels]。*

### 7. tf.image.encode_jpeg
#### 函数：
```
tf.image.encode_jpeg(image,
		format='',
		quality=95,
		progressive=False,
		optimize_size=False,
		chroma_downsampling=True,
		density_unit='in',
		x_density=300,
		y_density=300,
		xmp_metadata='',
		name=None)
```
#### 功能：
编码为 JPEG/JPG 图片。<br>
#### 参数：
`image` `uint8`型的`tensor`。3-D数组 [height, width, channels]。<br>
`format` string型，默认为`''`，表示为采用图片通道数。可选参数有: "", "grayscale", "rgb"。<br>
`quality` int型，默认为`95`。表示压缩质量，取值范围为0-100．<br>
`progressive` bool型，默认为`False`。若为`True`，则创建一个渐进加载的 JPEG/JPG 图像（由粗糙到精细）。<br>
`optimize_size` bool型，默认为`False`。若为`True`，则采用 CPU/ARM　减小大小而不影响质量。<br>
`chroma_downsampling=True` bool型，默认为`True`。<br>
`density_unit` string型，默认为`in`，可选参数有: 'in', 'cm'。用于制定`x_density`和`y_density`的单位: 像素点/英寸('in'),　像素点/厘米('cm')。<br>
`x_density` int型，默认为`300`。水平方向上每单位像素点数。<br>
`y_density` int型，默认为`300`。竖直方向上每单位像素点数。<br>
`xmp_metadata` string型，默认为`''`。若不为空，在图片头部嵌入此xmp。<br>
`name` 操作名（可选）。

### 8.tf.image.encode_png
#### 函数：
```
tf.image.encode_png(image,
		compression=-1,
		name=None)
```
#### 功能：
编码为 JPEG/JPG 图片。<br>
**参数：** `image` `uint8`型或`uint16`型的`tensor`。3-D数组 [height, width, channels]。<br>
`compression` int型，默认为`-1`。压缩级别。<br>
`name` 操作名（可选）。


## 二.图像尺寸变换
### 1. tf.image.convert_image_dtype  (这个以后放到图像变换里！！！！)
#### 函数：
```
tf.image.convert_image_dtype(image,
			dtype,
			saturate=False,
			name=None)
```
#### 功能：
将解码得到的图片`uint8`型的`tensor`转化为其他类型`DType`。<br>
#### 参数：
`image``uint8`型`tensor`。<br>
`dtype`。需要转化的类型，如`tf.float32`，则数值将被归一化在[0,1]之间。<br>
`saturate`bool型，默认为`False`。<br>
`name`操作名（可选）。

**注意：resize后返回的图片`tensor`类型为`float32`。**

### 1. tf.image.resize / tf.image.resize_images
#### 函数： 
```
tf.image.resize(images,
	size,
	method=ResizeMethod.BILINEAR,
	align_corners=False,
	preserve_aspect_ratio=False)
或
tf.image.resize_images(images,
	size,
	method=ResizeMethod.BILINEAR,
	align_corners=False,
	preserve_aspect_ratio=False)
```
#### 功能：
将图片尺寸变换为所需要的`size`。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸: [new_height, new_width]。<br>
`method` 插值方法。可选参数有: ResizeMethod.BILINEAR, ResizeMethod.NEAREST_NEIGHBOR, ResizeMethod.BICUBIC, ResizeMethod.AREA。


### 2. tf.image.resize_area
#### 函数：
```
tf.image.resize_area(images,
	size,
	align_corners=False,
	name=None)
```
#### 功能：
采用面积插值法将图片尺寸变换为所需要的`size`。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸: [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。

### 3. tf.image.resize_bicubic
#### 函数：
```
tf.image.resize_bicubic(images,
	size,
	align_corners=False,
	name=None)
```
#### 功能：
采用双三次插值法将图片尺寸变换为所需要的`size`。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸: [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。

### 4. tf.image.resize_billinear
#### 函数： 
```
tf.image.resize_billinear(images,
	size,
	align_corners=False,
	name=None)
```
#### 功能：
采用双线性插值法将图片尺寸变换为所需要的`size`。<br>
#### 参数： 
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸: [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。

### 5. tf.image.resize_nearest_neighbor
#### 函数：
```
tf.image.resize_nearest_neighbor(images,
	size,
	align_corners=False,
	name=None)
```
#### 功能：
采用最近邻插值法将图片尺寸变换为所需要的`size`，返回的`tensor`类型与输入图片相同。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸: [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。


### 6. tf.image.resie_image_with_pad
#### 函数：
```
tf.image.resize_image_with_pad(image,
			target_height,
			target_width,
			method=ResizeMethod.BILINEAR)
```
#### 功能：
将
