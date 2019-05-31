# tensorflow
*　tensorflow-1.13
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
tf.image.decode_jpeg(
		contents, 
		channels=0, 
		ratio=1, 
		fancy_upscaling=True,
		try_recover_truncated=False,
		acceptable_fraction=1,
		dct_method='',
		name=None
)
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
tf.image.decode_and_crop_jpeg(
		contents, 
		crop_window,
		channels=0, 
		ratio=1, 
		fancy_upscaling=True,
		try_recover_truncated=False,
		acceptable_fraction=1,
		dct_method='',
		name=None
)
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
tf.image.decode_png(
		contents,
		channels=0,
		dtype=tf.dtypes.uint8,
		name=None
)
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
tf.image.decode_bmp(
		contents,
		channels=0,
		name=None
)
```
#### 功能：
 将 PNG 图片解码为`uint8`类型的`tensor`。<br>
#### 参数：
`channels` int型，默认为`0`。0:使用输入图片的通道数； 1:输出为灰度图； 3:输出为RGB图。<br>
`name`操作名（可选）。

### 5.　tf.image.decode_gif
#### 函数：
```
tf.image.decode_gif(
		contents,
		name=None
)
```
#### 功能：
将 GIF 图像的第一帧解码为`uint8`类型的`tensor`。<br>
#### 参数：
`name` 操作名（可选）。

### 6.　tf.image.decode_image
#### 函数：
```
tf.image.decode_image(
		contents,
		channels=None,
		dtype=tf.dtypes.uint8,
		name=None
)
```
#### 功能：
将 BMP/GIF/JPEG/JPG/PNG 图片采用合适操作转换为`dtype`型的`tensor`。<br>
*注：解码 GIF 图片返回的是4-D数组 [num_frames, height, width, 3]，解码其他类型图片返回的是3-D数组 [height, width, num_channels]。*

### 7. tf.image.encode_jpeg
#### 函数：
```
tf.image.encode_jpeg(
		image,
		format='',
		quality=95,
		progressive=False,
		optimize_size=False,
		chroma_downsampling=True,
		density_unit='in',
		x_density=300,
		y_density=300,
		xmp_metadata='',
		name=None
)
```
#### 功能：
编码为 JPEG/JPG 图片。<br>
#### 参数：
`image` `uint8`型的`tensor`。3-D数组 [height, width, channels]。<br>
`format` string型，默认为`''`，表示为采用图片通道数。可选参数有: '', 'grayscale', 'rgb'。<br>
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
tf.image.encode_png(
		image,
		compression=-1,
		name=None
)
```
#### 功能：
编码为 JPEG/JPG 图片。<br>
**参数：** `image` `uint8`型或`uint16`型的`tensor`。3-D数组 [height, width, channels]。<br>
`compression` int型，默认为`-1`。压缩级别。<br>
`name` 操作名（可选）。


## 二.图像格式变换
### 1. tf.image.convert_image_dtype  
#### 函数：
```
tf.image.convert_image_dtype(
			image,
			dtype,
			saturate=False,
			name=None
)
```
#### 功能：
将解码得到的图片`uint8`型的`tensor`转化为其他类型`DType`。<br>
#### 参数：
`image` `uint8`型`tensor`。<br>
`dtype` 需要转化的类型，如`tf.float32`，则数值将被归一化在[0,1]之间。<br>
`saturate` bool型，默认为`False`。<br>
`name` 操作名（可选）。

### 2. tf.image.rgb_to_grayscale
#### 函数：
```
tf.image.rgb_to_grayscale(
    images,
    name=None
)
```
####　功能：
将 RGB 图像转换为灰度图，输出的`tensor`的数据类型与输入的`images`一致。
#### 参数：
`images`　RGB图像`tensor`，数值取值范围为 [0,1]。<br>
`name` 操作名（可选）。

### 3. tf.image.rgb_to_hsv
#### 函数：
```
tf.image.rgb_to_hsv(
    images,
    name=None
)
```
####　功能：
将 RGB 图像转换为 HSV 图像，输出的`tensor`的形状与输入的`images`一致。<br>
输出的 `output[..., 0]`代表色度，　`output[..., 1]`代表饱和度，`output[..., 0]`代表数值。<br>
色度为0代表红色，色度为1/3代表绿色，色度为2/3代表蓝色。
#### 参数：
`images`　RGB图像`tensor`，数据类型需要为以下类型其中之一：half, bfloat16, float32, float64。<br>
`name` 操作名（可选）。

### 4. tf.image.rgb_to_yuv
#### 函数：
```
tf.image.rgb_to_yuv(images)

```
####　功能：
将 RGB 图像转换为 HSV 图像，输出的`tensor`的形状与输入的`images`一致。
#### 参数：
`images`　RGB图像`tensor`，数值取值范围为 [0,1]。<br>


### 5. tf.image.rgb_to_yiq
#### 函数：
```
tf.image.rgb_to_yiq(images)

```
####　功能：
将 RGB 图像转换为 YIQ 图像，输出的`tensor`的形状与输入的`images`一致。
#### 参数：
`images`　RGB图像`tensor`，数值取值范围为 [0,1]。<br>

### 6. tf.image.grayscale_to_rgb
#### 函数：
```
tf.image.grayscale_to_rgb(
    images,
    name=None
)
```
####　功能：
将灰度图转换为 RGB 图像，输出的`tensor`的数据类型与输入的`images`一致。
#### 参数：
`images`　灰度图`tensor`。<br>
`name` 操作名（可选）。


### 7. tf.image.hsv_to_rgb
#### 函数：
```
tf.image.hsv_to_rgb(
    images,
    name=None
)
```
####　功能：
将 HSV 图像转换为 RGB 图像，输出的`tensor`的形状与输入的`images`一致。
#### 参数：
`images`　HSV 图像`tensor`，数值取值范围为 [0,1]。<br>
`name` 操作名（可选）。


### 8. tf.image.yuv_to_rgb
#### 函数：
```
tf.image.hyuv_to_rgb(images)
```
####　功能：
将 YUV 图像转换为 RGB 图像，输出的`tensor`的形状与输入的`images`一致，Y分量取值范围为 [0,1]，U,V分量的取值范围为 [-0.5, 0.5]。
#### 参数：
`images`　YUV 图像`tensor`。<br>



### 9. tf.image.yiq_to_rgb
#### 函数：
```
tf.image.yiq_to_rgb(images)
```
####　功能：
将 YUV 图像转换为 RGB 图像，输出的`tensor`的形状与输入的`images`一致，Y分量取值范围为 [0,1]，I分量的取值范围为 [-0.5957, 0.5957]，Q分量的取值范围为 [-0.5226,0.5226]。
#### 参数：
`images`　YIQ 图像`tensor`。<br>


## 三.图像尺寸变换

**注意：resize后返回的图片`tensor`的数据类型为`float32`。**

### 1. tf.image.resize / tf.image.resize_images
#### 函数： 
```
tf.image.resize(
	images,
	size,
	method=ResizeMethod.BILINEAR,
	align_corners=False,
	preserve_aspect_ratio=False
)
或
tf.image.resize_images(
	images,
	size,
	method=ResizeMethod.BILINEAR,
	align_corners=False,
	preserve_aspect_ratio=False
)
```
#### 功能：
将图片尺寸变换为所需要的`size`。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸: size = [new_height, new_width]。<br>
`method` 插值方法，可选参数有: ResizeMethod.BILINEAR, ResizeMethod.NEAREST_NEIGHBOR, ResizeMethod.BICUBIC, ResizeMethod.AREA。


### 2. tf.image.resize_area
#### 函数：
```
tf.image.resize_area(
	images,
	size,
	align_corners=False,
	name=None
)
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
tf.image.resize_bicubic(
	images,
	size,
	align_corners=False,
	name=None
)
```
#### 功能：
采用双三次插值法将图片尺寸变换为所需要的`size`。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸:  size = [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。

### 4. tf.image.resize_billinear
#### 函数： 
```
tf.image.resize_billinear(
	images,
	size,
	align_corners=False,
	name=None
)
```
#### 功能：
采用双线性插值法将图片尺寸变换为所需要的`size`。<br>
#### 参数： 
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸:  size = [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。

### 5. tf.image.resize_nearest_neighbor
#### 函数：
```
tf.image.resize_nearest_neighbor(
	images,
	size,
	align_corners=False,
	name=None
)
```
#### 功能：
采用最近邻插值法将图片尺寸变换为所需要的`size`。<br>
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`size`　1-D的`int32`型的`tensor`。变换后的图片尺寸:  size = [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name`操作名（可选）。


### 6. tf.image.resie_image_with_pad
#### 函数：
```
tf.image.resize_image_with_pad(
			image,
			target_height,
			target_width,
			method=ResizeMethod.BILINEAR
)
```
#### 功能：
对图片进行尺寸变换并填充为目标尺寸。<br>
若目标尺寸的长宽比与原图的长宽比一致，则进行不失真的尺寸变换；若目标尺寸的长宽比与原图的长宽比不一致，则按照原图长宽比进行图片尺寸变换后对剩余位置填充0。<br>
返回形状为与输入`image`一致的`float`型`tensor`。若`image`为4-D，则返回的`tensor`形状为 [batch, new_height, new_width, channels]；若`image`为3-D，则返回的`tensor｀形状为 [new_height, new_width, channels]。
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>
`target_height` 目标高度。<br>
`target_width`　目标宽度。<br>
`method`　插值方法，可选参数有：ResizeMethod.BILINEAR, ResizeMethod.NEAREST_NEIGHBOR, ResizeMethod.BICUBIC, ResizeMethod.AREA。

#### 7. tf.image.resize_image_with_crop_or_pad
#### 函数：
```
tf.image.resize_image_with_crop_or_pad(
			image,
			target_height,
			target_width
)
```
#### 功能：
对图片进行裁剪或/和填充为目标尺寸。<br>
若原图的宽度（高度）大于目标宽度（高度），则在宽度（高度）维度上进行中心裁剪；原图的宽度（高度）大于目标宽度（高度），则在宽度（高度）维度上填充0。<br>
返回的形状为与输入`image`一致的`float`型`tensor`。若`image`为4-D，则返回的`tensor`形状为 [batch, new_height, new_width, channels]；若`image`为3-D，则返回的`tensor｀形状为 [new_height, new_width, channels]。
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>
`target_height` 目标高度。<br>
`target_width`　目标宽度。<br>

## 四.图像裁剪


### 1. tf.image.central_crop
#### 函数：
```
tf.iamge.central_crop(
	image,
	central_fraction
)
```
#### 功能：
对图片进行中心裁剪，返回3-D/4-D的`float`型`tensor`。
#### 参数：
`image` 4-D的`float`型`tensor`，形状为 [batch, height, width, channels]；或3-D的`float`型`tensor`，形状为　[height, width, channels]。<br>
`central_fraction`　float型，裁剪的尺寸比例，取值范围 (0,1]。

### 2. tf.image.random_crop
#### 函数：
```
tf.image.random_crop(
		value,
		size,
		seed=None,
		name=None
)
```
#### 功能：
将图片随机裁剪为给定尺寸，返回的`tensor`数据类型与`image`相同。
#### 参数：
`image`　输入的图片`tensor`。<br>
`size`　1-D的`tensor`，需要裁剪的尺寸大小，size = [crop_height, crop_width, 3]。<br>
`seed`　Python整数，用于创建随机种子。<br>
`name`操作名（可选）。

### 3. tf.image.crop_to_bounding_box
####　函数：
```
tf.image.crop_to_bounding_box(
		image,
		offset_height,
		offset_width,
		target_height,
		target_width,
)
```
#### 功能：
对图片的框中区域进行裁剪。<br>
返回形状为与输入`image`一致的`float`型`tensor`。若`image`为4-D，则返回的`tensor`形状为 [batch, target_height, target_width, channels]；若`image`为3-D，则返回的`tensor｀形状为 [target_height, target_width, channels]。
#### 参数：
`image` 4-D的`float`型`tensor`，形状为 [batch, height, width, channels]；或3-D的`float`型`tensor`，形状为　[height, width, channels]。<br>
`offset_height` 裁剪部分的左上角纵坐标。<br>
`offset_width` 裁剪部分的左上角横坐标。<br>
`target_height` 裁剪部分的高度。<br>
`target_width`　裁剪部分的宽度。

### 4. tf.image.crop_and_resize
####　函数：
```
tf.image.crop_and_resize(
		image, 
		boxes,
		box_ind,
		crop_size,
		method='bilinear',
		extrapolation_value=0,
		name=None
)
```
#### 功能：
对图片进行裁剪后再将其变换为给定尺寸。返回4-D的 [num_boxes, crop_height, crop_width, depth]
#### 函数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`boxes` `float32`型的2-D`tensor`，形状为 [num_boxes,4]。`tensor`的第 i 行制定`box_ind[i]`图像中框的坐标 [y1, x1, y2, x2]，并且需要归一化至 [0, 1] 之间。允许 y1>y2　的处理，这种情况下裁剪的图片为原始图片的上下翻转，宽度维度的处理方式相似。另外， [0, 1] 范围之外的标准化坐标是允许的，在这种情况下，使用`extrapolation_value`外推输入图像值.<br>
`box_ind`　`int32`型的1-D`tensor`，形状为 [num_boxes]，取值范围为 [0, batch)。`box_ind[i]`指定第 i 个框要引用的图像。　<br>
`crop_size`　`int32`型的1-D`tensor`，crop_size = [crop_height, crop_width]。<br>
`method`　resize过程中的插值方法，默认为`'bilinear'`。可选参数有：'bilinear', 'nearest'。<br>
`extrapolation_value` `float`型，用于外推图像值。<br>
`name`操作名（可选）。

### 5. tf.image.extract_glimpse
#### 函数：
```
tf.image.extract_glimpse(
    input,
    size,
    offsets,
    centered=True,
    normalized=True,
    uniform_noise=True,
    name=None
)
```
#### 功能：
依据裁剪框的坐标生成一系列裁剪后的图像，若裁剪框与输入图片仅有部分重合，则不重合部分填充随机噪声。。<br>
返回`float32`型的4-D`tensor`，形状为 [batch_size, glimpse_height, glimpse_width, channels]
#### 参数：
`input` 4-D的`float`型`tensor`，形状为 [batch_size, height, width, channels]。<br>
`size`　1-D的`int`型`tensor`，裁剪框的尺寸。size = [glimpse_height, glimpse_width]。 　　
`offsets` 2-D的`float32`型`tensor`，形状为 [batch_size, 2]，表示每个裁剪框的中心坐标 [y,x]。<br>
`centered` `bool`型，默认为`True`。若为`True`，表示 (0,0)坐标为输入图片的中心；若为`False`，表示 (0,0)坐标为输入图片的的左上角。<br>
`normalized`　`bool`型，默认为`True`。指示`offsets`坐标是否被归一化。<br>
`uniform_nose`　`bool`型，默认为`True`。指示填充的随机噪声是否为平均分布，否则为高斯分布。<br>
`name`操作名（可选）。

### tf.image.pad_to_bounding_box
#### 函数：
```
tf.image.pad_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
)
```
#### 功能：
给图片填充0至目标尺寸。<br>
返回形状为与输入`image`一致的`float`型`tensor`。若`image`为4-D，则返回的`tensor`形状为 [batch, target_height, target_width, channels]；若`image`为3-D，则返回的`tensor｀形状为 [target_height, target_width, channels]。
#### 参数：
`image` 4-D的`float`型`tensor`，形状为 [batch, height, width, channels]；或3-D的`float`型`tensor`，形状为　[height, width, channels]。<br>
`offset_height` 在图片上方填充的行数。<br>
`offset_width`　在图片左边填充的列数。<br>
`target_height` 输出图像的高度。<br>
`target_width`　输出图像的宽度。<br>
