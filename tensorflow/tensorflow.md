# tensorflow
* tensorflow-1.13
tensorflow中采用`tf.image`进行图像处理
```
import tensorflow as tf
contents = tf.gfile.FastGFile(image_path,'r').read()  # 读取图片
```

读取得到的`contents`为`string`型的 0-D `tensor`，需要进行解码转换为tensorflow可用的数据类型。

------

## tensorflow 图像处理方法
* 一. 图像编码与解码
* 二. 图像格式变换
* 三. 图像尺寸变化
* 四. 图像裁剪
* 五. 图像翻转与旋转
* 六. 图像色彩变换
* 七. 图像其他操作

-----

## 一. 图像编码与解码
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
`dct_method` string型，默认为`""`，表示为选择系统默认方法。解压缩方法，可选参数有："INTEGER_FAST",  "INTEGER_ACCURATE"。<br>
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
`dtype` tf.DType型，默认为`tf.uint8`。可选参数有：tf.uint8，tf.uint16。<br>
`name` 操作名（可选）。

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
`image` uint8型的`tensor`。3-D数组 [height, width, channels]。<br>
`format` string型，默认为`''`，表示为采用图片通道数。可选参数有: '', 'grayscale', 'rgb'。<br>
`quality` int型，默认为`95`。表示压缩质量，取值范围为0-100．<br>
`progressive` bool型，默认为`False`。若为`True`，则创建一个渐进加载的 JPEG/JPG 图像（由粗糙到精细）。<br>
`optimize_size` bool型，默认为`False`。若为`True`，则采用 CPU/ARM　减小大小而不影响质量。<br>
`chroma_downsampling=True` bool型，默认为`True`。<br>
`density_unit` string型，默认为`in`，可选参数有: 'in', 'cm'。用于制定`x_density`和`y_density`的单位: 像素点/英寸('in'),　像素点/厘米('cm')。<br>
`x_density` int型，默认为`300`。水平方向上每单位像素点数。<br>
`y_density` int型，默认为`300`。垂直方向上每单位像素点数。<br>
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
#### 参数：
`image` uint8型或uint16型的`tensor`。3-D数组 [height, width, channels]。<br>
`compression` int型，默认为`-1`。压缩级别。<br>
`name` 操作名（可选）。


## 二. 图像格式变换
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
`image` uint8型`tensor`。<br>
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
#### 功能：
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
#### 功能：
将 RGB 图像转换为 HSV 图像，输出的`tensor`的形状与输入的`images`一致。<br>
输出的 `output[..., 0]`代表色度，`output[..., 1]`代表饱和度，`output[..., 0]`代表数值。<br>
色度为0代表红色，色度为1/3代表绿色，色度为2/3代表蓝色。
#### 参数：
`images`　RGB图像`tensor`，数据类型需要为以下类型其中之一：half, bfloat16, float32, float64。<br>
`name` 操作名（可选）。

### 4. tf.image.rgb_to_yuv
#### 函数：
```
tf.image.rgb_to_yuv(images)

```
#### 功能：
将 RGB 图像转换为 HSV 图像，输出的`tensor`的形状与输入的`images`一致。
#### 参数：
`images`　RGB图像`tensor`，数值取值范围为 [0,1]。<br>


### 5. tf.image.rgb_to_yiq
#### 函数：
```
tf.image.rgb_to_yiq(images)

```
#### 功能：
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
#### 功能：
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
#### 功能：
将 HSV 图像转换为 RGB 图像，输出的`tensor`的形状与输入的`images`一致。
#### 参数：
`images` HSV 图像`tensor`，数值取值范围为 [0,1]。<br>
`name` 操作名（可选）。


### 8. tf.image.yuv_to_rgb
#### 函数：
```
tf.image.hyuv_to_rgb(images)
```
#### 功能：
将 YUV 图像转换为 RGB 图像，输出的`tensor`的形状与输入的`images`一致，Y分量取值范围为 [0,1]，U,V分量的取值范围为 [-0.5, 0.5]。
#### 参数：
`images`　YUV 图像`tensor`。<br>


### 9. tf.image.yiq_to_rgb
#### 函数：
```
tf.image.yiq_to_rgb(images)
```
#### 功能：
将 YIQ 图像转换为 RGB 图像，输出的`tensor`的形状与输入的`images`一致，Y分量取值范围为 [0,1]，I分量的取值范围为 [-0.5957, 0.5957]，Q分量的取值范围为 [-0.5226,0.5226]。
#### 参数：
`images`　YIQ 图像`tensor`。<br>


## 三. 图像尺寸变换

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
`size`　1-D的int32型的`tensor`。变换后的图片尺寸: size = [new_height, new_width]。<br>
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
`size`　1-D的int32型的`tensor`。变换后的图片尺寸: [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name` 操作名（可选）。

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
`size`　1-D的int32型的`tensor`。变换后的图片尺寸:  size = [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name` 操作名（可选）。

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
`size`　1-D的int32型的`tensor`。变换后的图片尺寸:  size = [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name` 操作名（可选）。

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
`size`　1-D的int32型的`tensor`。变换后的图片尺寸:  size = [new_height, new_width]。<br>
`align_corners` bool型，默认为`False`。若为`True`，则对输入和输出`tensor`的四个角落的中心点保留像素值。<br>
`name` 操作名（可选）。


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
返回形状为与输入`image`一致的`float`型`tensor`。若`image`为4-D，则返回的`tensor`形状为 [batch, new_height, new_width, channels]；若`image`为3-D，则返回的`tensor`形状为 [new_height, new_width, channels]。
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



## 四. 图像裁剪


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
`image` 4-D的float型`tensor`，形状为 [batch, height, width, channels]；或3-D的float型`tensor`，形状为　[height, width, channels]。<br>
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
`name` 操作名（可选）。

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
`image` 4-D的float型`tensor`，形状为 [batch, height, width, channels]；或3-D的float型`tensor`，形状为　[height, width, channels]。<br>
`offset_height` 裁剪部分的左上角纵坐标<br>
`offset_width` 裁剪部分的左上角横坐标<br>
`target_height` 裁剪部分的高度<br>
`target_width`　裁剪部分的宽度

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
对图片进行裁剪后再将其变换为给定尺寸。返回4-D的`tensor`，形状为 [num_boxes, crop_height, crop_width, depth]
#### 函数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]。数据类型需要为以下类型其中之一：int8, uint8, int16, uint16, int32, int64, half, float32, float64。<br>
`boxes` float32型的2-D`tensor`，形状为 [num_boxes,4]。`tensor`的第 i 行制定`box_ind[i]`图像中框的坐标 [y1, x1, y2, x2]，并且需要归一化至 [0, 1] 之间。允许 y1>y2　的处理，这种情况下裁剪的图片为原始图片的上下翻转，宽度维度的处理方式相似。另外， [0, 1] 范围之外的标准化坐标是允许的，在这种情况下，使用`extrapolation_value`外推输入图像值.<br>
`box_ind`　int32型的1-D`tensor`，形状为 [num_boxes]，取值范围为 [0, batch)。`box_ind[i]`指定第 i 个框要引用的图像　<br>
`crop_size`　int32型的1-D`tensor`，输出图像的形状 crop_size = [crop_height, crop_width]<br>
`method`　resize过程中的插值方法，默认为`'bilinear'`。可选参数有：'bilinear', 'nearest'<br>
`extrapolation_value` float型，用于外推图像值。<br>
`name` 操作名（可选）。

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
`input` 4-D的float型`tensor`，形状为 [batch_size, height, width, channels]。<br>
`size`　1-D的int型`tensor`，裁剪框的尺寸。size = [glimpse_height, glimpse_width]。 　　
`offsets` 2-D的float32型`tensor`，形状为 [batch_size, 2]，表示每个裁剪框的中心坐标 [y,x]。<br>
`centered` `bool`型，默认为`True`。若为`True`，表示 (0,0)坐标为输入图片的中心；若为`False`，表示 (0,0)坐标为输入图片的的左上角。<br>
`normalized`　`bool`型，默认为`True`。指示`offsets`坐标是否被归一化。<br>
`uniform_nose`　`bool`型，默认为`True`。指示填充的随机噪声是否为平均分布，否则为高斯分布。<br>
`name`操作名（可选）。


### 6. tf.image.extract_image_patches
#### 函数：
```
tf.image.extract_image_patches(
    images,
    ksizes,
    strides,
    rates,
    padding,
    name=None
)
```
#### 功能：
从图片`images`中提取一系列`patches`，并把他们放置于 "depth"　维度上。
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, in_rows, in_cols, depth]。数据类型需要为以下类型其中之一： float32, float64, int32, uint8, int16, int8, int64, bfloat16, uint16, half, uint32, uint64。<br>
`ksizes`　`int`型列表，滑动窗口的大小，ksize = [1, ksize_rows, ksize_cols, 1]。<br>
`strides` `int`型列表，滑动窗口的步长，strides = [1, strides_rows, strides_cols, 1]。<br>
`rates`　`int`型列表，在原始图像的一块`patch`中，隔多少像素点，取一个有效像素点。rates = [1, rates_rows, rates_cols, 1]。<br>
`padding` `string`型，填充方式。可选参数有： 'SAME', 'VALID'。<br>
`name`操作名（可选）。


### 7. tf.image.pad_to_bounding_box
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

## 五. 图像翻转和旋转
### 1. tf.image.flip_left_right
####　函数：
```
tf.image.flip_left_right(images)
```
#### 功能：
将图片水平翻转，输出的`tensor`的形状和数据类型与输入的`image`相同。
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>


### 2. tf.image.flip_up_down
####　函数：
```
tf.image.flip_up_down(images)
```
#### 功能：
将图片垂直翻转，输出的`tensor`的形状和数据类型与输入的`image`相同。
#### 参数：
`images` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>

### 3. tf.image.random_flip_left_right
#### 函数
```
tf.image.random_flip_left_right(
		image,
		seed=None
)
```
#### 功能：
以0.5的概率随机进行图片水平翻转。
#### 参数：
`image` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>
`seed`　Python整数，用于创建随机种子。


### 4. tf.image.random_flip_up_down
#### 函数
```
tf.image.random_flip_up_down(
		image,
		seed=None
)
```
#### 功能：
以0.5的概率随机进行图片垂直翻转。
#### 参数：
`image` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>
`seed`　Python整数，用于创建随机种子。

### 5. tf.image.rot90
#### 函数
```
tf.image.rot90(
	image,
	k=1,
	name=None
)

```
#### 功能：
将图片逆时针旋转90度。
#### 参数：
`image` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>
`k` 整数，图片逆时针旋转90度的次数。<br>
`name` 操作名（可选）。

## 六. 图像色彩变换
### 1. tf.image.adjust_brightness
#### 函数：
```
tf.image.adjust_brightness(
        image,
        delta
)
```
#### 功能：
调整图片（RGB图像或者灰度图）的亮度。
#### 参数：
`image` 图像`tensor`。<br>
`delta` float型，调整图像亮度，取值范围 [0, 1)。

### 2. tf.image.adjust_contrast
#### 函数：
```
tf.image.adjust_contrast(
        images,
        contrast_factor
)
```
#### 功能：
调整图片（RGB图像或者灰度图）的对比度。
#### 参数：
`images` 图像`tensor`<br>
`contrast_factor` float型，调整图像对比度。对于每个channel的像素x，调整后的像素值为`(x-mean)xcontrast_factor+mean`。


### 3. tf.image.adjust_hue
#### 函数：
```
tf.image.adjust_hue(
        image,
        delta,
        name=None
)
```
#### 功能：
调整图片（RGB图像）的色度。
#### 参数：
`image` 图像`tensor`。<br>
`delta` float型，调整图像色度，取值范围 [-1, 1]。<br>
`name` 操作名（可选）。

### 4. tf.image.adjust_saturation
#### 函数：
```
tf.image.adjust_saturation(
        image,
        saturation_factor,
        name=None
)
```
#### 功能：
调整图片（RGB图像）的饱和度。
#### 参数：
`image` 图像`tensor`。<br>
`delta` floa`型，调整图像饱和度。<br>
`name` 操作名（可选）。

### 5. tf.image.adjust_gamma
#### 函数：
```
tf.image.adjust_gamma(
        image,
        gamma=1,
        gain=1
)
```
#### 功能：
对图像进行gamma校正。
#### 参数：
`image` 图像`tensor`。<br>
`gamma` `scalar`或`tensor`，对像素值归一化后进行校正`out=in**gamma`。<br>
`gain` `scalar`或`tensor`，常数乘子。

### 6. tf.image.adjust_jpeg_quality
#### 函数：
```
tf.image.adjust_jpeg_quality(
        image,
        jpeg_quality,
        name-None
)
```
#### 功能：
调整 RGB 图像编码质量。
#### 参数：
`image` 图像`tensor`。<br>
`jpeg_quality` int型，JPEG图像编码质量，取值范围为 [0, 100]。<br>
`name` 操作名（可选）。

### 7. tf.image.random_brightness
#### 函数：
```
tf.image.random_brightness(
        image,
        max_delta,
        seed=None
)
```
#### 功能：
随机调整图片（RGB图像或者灰度图）的亮度。
#### 参数：
`image` 图像`tensor`。<br>
`max_delta` float型，调整图像亮度，调整值在 [-max_delta, max_delta]范围里随机选择。<br>
`seed`　Python整数，用于创建随机种子。<br>


### 8. tf.image.random_contrast
#### 函数：
```
tf.image.random_contrast(
        images,
        lower,
        upper,
        seed=None
)
```
#### 功能：
随机调整图片（RGB图像）的对比度。
#### 参数：
`images` 图像`tensor`。<br>
`lower` float型，对比度因子下限。<br>
`upper` float型，对比度因子上限。<br>
`seed`　Python整数，用于创建随机种子。<br>


### 9. tf.image.random_hue
#### 函数：
```
tf.image.random_hue(
        image,
        max_delta,
        seed=None
)
```
#### 功能：
随机调整图片（RGB图像）的色度。
#### 参数：
`image` 图像`tensor`。<br>
`max_delta` float型，调整图像色度，取值范围 [0, 0.5]，调整值在 [-max_delta, max_delta]范围里随机选择。<br>
`seed`　Python整数，用于创建随机种子。<br>

### 10. tf.image.random_saturation
#### 函数：
```
tf.image.random_saturation(
        image,
        lower,
        upper,
        seed=None
)
```
#### 功能：
随机调整图片（RGB图像）的饱和度。
#### 参数：
`image` 图像`tensor`<br>
`lower` float型，饱和度因子下限<br>
`upper` float型，饱和度因子上限<br>
`seed`　Python整数，用于创建随机种子。<br>


### 11. tf.image.random_jpeg_quality
#### 函数：
```
tf.image.adjust_jpeg_quality(
        image,
        min_jpeg_quality,
        max_jpeg_quality,
        name-None
)
```
#### 功能：
随机调整 RGB 图像编码质量。
#### 参数：
`image` 图像`tensor`。<br>
`min_jpeg_quality` int型，JPEG图像编码质量下限，取值范围为 [0, 100]。<br>
`max_jpeg_quality` int型，JPEG图像编码质量上限，取值范围为 [0, 100]，且要大于 `min_jpeg_quality`。<br>

## 七. 图像其他操作
### 1. tf.image.image_gradients
#### 函数：
```
tf.image.image_gradients(image)
```
#### 功能：
计算图像每个channel的水平和垂直梯度（一阶有限差分），返回一对`tensor`(dx, dy)，每个输出`tensor`都和输入图像`tensor`的形状一致。
#### 参数：
`image` 4-D的`tensor`，形状为 [batch_size, height, width, depth]。<br>

### 2. tf.image.sobel_edges
#### 函数：
```
tf.image.sobel_edges(image)
```
#### 功能：
计算图像的sobel边缘映射张量，返回的`tensor`形状为 [batch_size, height, width, depth, 2]。
#### 参数
`image` 4-D的float32或`tensor`，形状为 [batch_size, height, width, depth]。<br>

### 3. tf.image.per_image_standardization
#### 函数：
```
tf.image.per_image_standardization(image)
```
#### 功能：
对图像的像素值进行标准化，返回与`image`形状一致的`tensor`。<br>
这个函数对图像的每个像素进行了操作 `(x-mean)/adjusted_stddev`，其中`mean`为所有像素的均值，`adjusted_stddev=max(stddev, 1.0/sqrt(image.NumElements()))`，`stddev`为所有像素的标准差，`image.NumElements()`为所有像素个数。
#### 参数：
`image` 3-D的`tensor`，形状为 [height, width, channels]。

### 4. tf.image.total_variation
#### 函数：
```
tf.image.total_variation(
            images,
            name=None
)
```
#### 功能：
计算一个或多个图像的总体变化量。<br>
总体变化量为输入图像中相邻像素值绝对差值的总和，可用于测量图像中有多少噪声量。
#### 参数：

`images` 4-D的`tensor`，形状为 [batch, height, width, channels]；或3-D的`tensor`，形状为　[height, width, channels]。<br>
`name` 操作名（可选）。

### 5. tf.image.psnr
#### 函数：
```
tf.image.psnr(
        a,
        b,
        max_val,
        name=None
)
```
#### 示例：
```
# Read images from file.
im1 = tf.decode_png('path/to/im1.png')
im2 = tf.decode_png('path/to/im2.png')
# Compute PSNR over tf.uint8 Tensors.
psnr1 = tf.image.psnr(im1, im2, max_val=255)

# Compute PSNR over tf.float32 Tensors.
im1 = tf.image.convert_image_dtype(im1, tf.float32)
im2 = tf.image.convert_image_dtype(im2, tf.float32)
psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
# psnr1 and psnr2 both have type tf.float32 and are almost equal.
```
#### 功能：
计算 a 和 b 的峰值信噪比，返回`tf.float32`型的`tensor`，形状为 [batch_size, 1]。<br>
#### 参数：
`a` 第一组图像。 <br>
`b` 第二组图像。 <br>
`max_val` 图像的动态范围(即最大允许值和最小允许值之间的差值)。<br>
`name` 操作名（可选）。

### 6. tf.image.ssim
#### 函数：
```
tf.image.ssim(
        img1,
        img2,
        max_val
)
```
#### 示例：
```
# Read images from file.
im1 = tf.decode_png('path/to/im1.png')
im2 = tf.decode_png('path/to/im2.png')
# Compute SSIM over tf.uint8 Tensors.
ssim1 = tf.image.ssim(im1, im2, max_val=255)

# Compute SSIM over tf.float32 Tensors.
im1 = tf.image.convert_image_dtype(im1, tf.float32)
im2 = tf.image.convert_image_dtype(im2, tf.float32)
ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
# ssim1 and ssim2 both have type tf.float32 and are almost equal.
```
#### 功能：
计算图像`img1`和`img2`之间的SSIM值（结构相似性）。返回的`tensor`形状为 (img1.shape[:-3], img2.shape[:-3])<br>
SSIM为图像质量评价指标，输出值为 (-1,1] 的float数，越大表示两幅图之间的差距越小。<br>
由于高斯滤波器尺寸的原因，输入图像的尺寸大小至少为11x11。
#### 参数：
`img1` 第一组图像。 <br>
`img2` 第二组图像。<br>
`max_val` 图像的动态范围(即最大允许值和最小允许值之间的差值)。<br>

### 7. tf.image.ssim_multiscale
#### 函数：
```
tf.image.ssim_multiscale(
        img1,
        img2,
        max_val,
        power_factors=_MSSSIM_WEIGHTS
)
```
#### 功能：
计算图像`img1`和`img2`之间的MS-SSIM值（多层级结构相似性）。返回的`tensor`形状为 (img1.shape[:-3], img2.shape[:-3])<br>
MS-SSIM为图像质量评价指标，输出值为 [0,1] 的float数，越大表示两幅图之间的差距越小。<br>
#### 参数：
`img1` 第一组图像。 <br>
`img2` 第二组图像。<br>
`max_val` 图像的动态范围(即最大允许值和最小允许值之间的差值)。<br>
`power_factor` 每个量表的权重都是可迭代的,所用的比例数是列表的长度，索引0是未缩放的分辨率的权重,下一个索引对应于当前图片缩小一半的权重,默认为(0.0448,0.2856,0.3001,0.2363,0.1333)。

### 8. tf.image.non_max_suppression
#### 函数：
```
tf.image.non_max_suppression(
            boxes, 
            scores,
            max_output_size,
            iou_threshold=0.5,
            score_threshold=float('-inf'),
            name=None
)
```
#### 功能：
以分数降序来选择边框的一个子集，移除与先前选择的边框具有很高IOU的边框。<br>
返回形状为 [M] 的1-D整型`tensor`，代表从`boxes`中选择的边框的索引，且 M<=max_output_size。

#### 参数：
`boxes` 2-D的float型`tensor`，形状为 [num_boxes, 4]，边框。<br>
`scores` 1-D的float型`tensor`，形状为 [num_boxes]，表示每个边框的置信度。<br>
`max_output_size` 整数，表示由NMS选出的边框的最大数量。<br>
`iou_threshold` float数，表示NMS中设置的IOU阈值，IOU高于此阈值的边框将被移除。<br>
`score_threshold` float数，表示根据置信度移除边框的阈值，置信度低于此阈值的边框将被移除。<br>
`name` 操作名（可选）。


### 9. tf.image.non_max_suppression_overlap
#### 函数：
```
tf.image.non_max_suppression_overlap(
            overlaps, 
            scores,
            max_output_size,
            overlap_threshold=0.5,
            score_threshold=float('-inf'),
            name=None
)
```
#### 功能：
以分数降序来选择边框的一个子集，移除与先前选择的边框具有很高IOU的边框。<br>
返回形状为 [M] 的1-D整型`tensor`，代表从`overlaps`中选择的框的索引，且 M<=max_output_size。

#### 参数：
`overlaps` 2-D的float型`tensor`，形状为 [num_boxes, num_boxes]，代表各个边框之间的重叠程度。<br>
`scores` 1-D的float型`tensor`，形状为 [num_boxes]，表示每个边框的置信度。<br>
`max_output_size` 整数，表示由NMS选出的边框的最大数量。<br>
`overlap_threshold` float数，表示NMS中设置的IOU阈值，高于此阈值的边框将被移除。<br>
`score_threshold` float数，表示根据置信度移除边框的阈值，置信度低于此阈值的边框将被移除。<br>
`name` 操作名（可选）。


### 10. tf.image.non_max_suppression_padded
#### 函数：
```
tf.image.non_max_suppression_padded(
            boxes, 
            scores,
            max_output_size,
            iou_threshold=0.5,
            score_threshold=float('-inf'),
            pad_to_max_output_size=False,
            name=None
)
```
#### 功能：
以分数降序来选择边框的一个子集，移除与先前选择的边框具有很高IOU的边框。<br>
当`pad_to_max_output_size=False`时，该函数与`tf.image.non_max_suppression`相同，返回形状为 [M] 的1-D整型`tensor`，代表从`boxes`中选择的边框的索引，且 M<=max_output_size。<br>
当`pad_to_max_output_size=True`时，返回形状为 [max_output_size] 的1-D整型`tensor`，对`M`之后的位置填充0元素。

#### 参数：
`boxes` 2-D的float型`tensor`，形状为 [num_boxes, 4]，边框。<br>
`scores` 1-D的float型`tensor`，形状为 [num_boxes]，表示每个边框的置信度。<br>
`max_output_size` 整数，表示由NMS选出的边框的最大数量。<br>
`iou_threshold` float数，表示NMS中设置的IOU阈值，IOU高于此阈值的边框将被移除。<br>
`score_threshold` float数，表示根据置信度移除边框的阈值，置信度低于此阈值的边框将被移除。<br>
`pad_to_max_output_size` bool型，默认为`False`，是否对输出的索引`tensor`填充0元素至长度为max_output_size。<br>
`name` 操作名（可选）。
