# MXNet
MXNet的图像处理API为`mxnet.image`
```
import mxnet as mx
```
MXNet处理的图像数据为`NDArray`格式，创建方式为`mx.nd.array`，注意不是`np.ndarray`格式。<br>


------------

## 目录
* 一. 图片处理操作
* 二. 图片迭代器
* 三. 图片增强操作

--------

## 一. 图片处理操作
### 1. mx.image.imread
#### 函数：
```
mx.image.imread(filename, **args, **kwargs)
```
#### 功能：
读取图片并将图片解码为`NDArray`格式，注意`imread`使用的是C++的opencv库而不是python的cv2库。
#### 参数：
`filename` string型，图片文件名。<br>
`flag` 0或1，默认为1。flag=1时为三通道输出，flag=0时为灰度图输出。<br>
`to_rgb` bool型，默认为True。若为True，输出通道格式为RGB；若为False，输出通道格式为BGR。

### 2. mx.image.imdecode
#### 函数：
```
mx.image,imdecode(buf, *args, **kwargs)
```
#### 功能：
将一张图片解码为`NDArray`格式，在这之前要先读取图片。注意`imread`使用的是C++的opencv库而不是python的cv2库。
#### 示例：
```
img = mxnet.image.imdecode(open("dog.jpg", "rb").read())
```
#### 参数：
`buf` string型的二值图片数据或numpy.ndarray数据。<br>
`flag` 0或1，默认为1。flag=1时为三通道输出，flag=0时为灰度图输出。<br>
`to_rgb` bool型，默认为True。若为True，输出通道格式为RGB；若为False，输出通道格式为BGR。


### 3. mx.image.imresize
#### 函数：
```
mx.image.imresize(src, w, h, *args, **kwargs)
```
#### 功能：
对图片进行尺寸变化。
#### 参数：
`src` NDArray型，图片数据。<br>
`w` int型，resize后的图片宽度。<br>
`h` int型，resize后的图片高度。<br>
`interp` 插值方法，int型，默认为1。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。

### 4. mx.image.scale_down
#### 函数：
```
mx.image.scale_down(src_size, size)
```
#### 功能：
若裁剪尺寸大于原图尺寸，则按比例缩放裁剪尺寸，返回缩放后的裁剪尺寸，格式为int型的元组。
#### 参数：
`src_size` int型的元组，原图尺寸。<br>
`size` int型的元组，原图尺寸。

### 5. mx.image.copyMakeBorder
#### 函数：
```
mx.image.copyMakeBorder(
            src=None,
            top=Null,
            bot=Null,
            left=Null,
            right=Null,
            type=Null,
            values=Null,
            out=None,
            name=None,
            **kwargs
)
```
#### 功能：
填充图片边界。
#### 参数：
`src` NDArray型，图片数据。<br>
`top` int型，上边界。<br>
`bot` int型，下边界。<br>
`left` int型，左边界。<br>
`right` int型，右边界。<br>
`type` int型，填充元素类型，默认为0，表示为cv2.BORDER_CONSTANT。<br>
`values` int型元组，填充像素值。<br>

### 6. mx.image.resize_short
#### 函数：
```
mx.image.resize_short(src, size, interp=2)
```
#### 功能：
通过设置较短的边进行图片尺寸变换，较长边按照比例相应变换。
#### 参数：
`src` NDArray型，图片数据。<br>
`size` int型，设置较短边的长度。<br>
`interp` 插值方法，int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。

### 7. mx.image.fixed_crop
#### 函数：
```
mx.image.fixed_crop(
            src,
            x0,
            y0,
            w,
            h,
            size=None,
            interp=2
)
```
#### 功能：
对图片进行裁剪，并resize到给定的尺寸。
#### 参数：
`src` NDArray型，图片数据。<br>
`x0` int型，裁剪区域的左上角横坐标。<br>
`y0` int型，裁剪区域的左上角纵坐标。<br>
`w` int型，裁剪区域的宽度。<br>
`h` int型，裁剪区域的高度。<br>
`size` (w,h)形式的元组，resize后的尺寸大小（可选）。<br>
`interp` 插值方法，int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。

### 8. mx.image.random_crop
#### 函数：
```
mx.image.random_crop(
            src,
            size,
            interp=2
)
```
#### 功能：
对图片进行随机裁剪。若图像`src`尺寸小于`size`，则对图像上采样。
#### 参数：
`src` NDArray型，图片数据。<br>
`size` (w,h)形式的元组，裁剪区域的尺寸大小。<br>
`interp` 插值方法，int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。

### 9. mx.image.center_crop
#### 函数：
```
mx.image.center_crop(
            src,
            size,
            interp=2
)
```
#### 功能：
对图片进行中心裁剪。若图像`src`尺寸小于`size`，则对图像上采样。<br>
返回值包括裁剪后的的`NDArray`型图像数据以及 (x,y,width,height) 形式的元组。
#### 示例：
```
>>> with open("flower.jpg", 'rb') as fp:
...     str_image = fp.read()
...
>>> image = mx.image.imdecode(str_image)
>>> image

>>> cropped_image, (x, y, width, height) = mx.image.center_crop(image, (1000, 500))
>>> cropped_image

>>> x, y, width, height
(1241, 910, 1000, 500)
```
#### 参数：
`src` NDArray型，图片数据。<br>
`size` (x,y,w,h)形式的元组，裁剪区域的尺寸大小。<br>
`interp` 插值方法，int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。

### 10. mx.image.color_normalize
#### 函数：
```
mx.image.color_normalize(
            src,
            mean,
            std=None
)
```
#### 功能：
图像像素值标准化。
#### 参数：
`src` NDArray型，图片数据。<br>
`mean` NDArray型，RGB通道均值。<br>
`std` NDArray型，RGB通道标准差。<br>

### 11. mx.image.random_size_crop
#### 函数：
```
mx.image.random_size_crop(
            src,
            size,
            area,
            ratio,
            interp=2,
            **kwargs
)
```
#### 功能：
按照`size`随机裁剪图片，之后再采用随机面积和长宽比进行resize。
#### 参数：
`src` NDArray型，图片数据。<br>
`size` (w,h)形式的元组，裁剪区域的尺寸大小。<br>
`area` (0,1]之间的float数或（float,float)型的元组。表示所使用的面积上下限，若为float型，则上限为1.0。<br>
`ratio` (float, float)型的元组，表示宽高比的上下限。<br>
`interp` 插值方法，int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。

--------------------


## 二. 图片迭代器
mxnet支持通过创建图片迭代器`mxnet.image.ImageIter`或`mxnet.image.ImageDetIter`（用于object detection任务），从`Record IO`或原始图片文件中读取图片。
### 1. class mx.image.ImageIter
```
class mxnet.image.ImageIter(batch_size, data_shape, label_width=1, path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None, shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, data_name='data', label_name='softmax_label', dtype='float32', last_batch_handle='pad', **kwargs)
```
#### 参数：
`batch_size` int型，每个batch的图片数量。<br>
`data_shape` (channels, height, width)格式的元组，目前仅支持RGB图像。<br>
`label_width` int型，默认为1。每张图片的label数量。<br>
`path_imgrec` str型，.rec文件路径。.rec文件可由im2rec.py生成。<br>
`path_imglist` str型，.lst文件路径。.lst文件可由im2rec.py生成。<br>
`imglist` list型，图片和label的列表。每个元素为一个列表 [imagelabel：float or list of loat, imagepath]。<br>
`path_root` str型，存放图片的文件夹路径。<br>
`path_imgidx` str型，存放图片索引文件的路径。当使用.rec文件进行partition和shuffle时需要。<br>
`shuffle` bool型，是否需要在每次迭代开始前进行shuffle。<br>
`part_index` int型，partition索引。<br>
`num_parts` int型，partition的数量。<br>
`data_name` str型，数据名称。<br>
`label_name` str型，label名称。<br>
`dtype` label数据类型，默认为float32。可选参数有 int32, int64, float64。<br>
`last_batch_handle` str型，处理最后一个batch的方式，默认为'pad'。可选参数有 'pad'：则最后一个batch数据由最开始的数据来填充；'discard'：则最后一个batch的数据被丢弃；'roll_over'：将最后一个batch的数据放到下一次迭代。<br>
`aug_list` 用于进行图片增强操作，详见三. 图片增强操作。
```
aug_list = mx.image.CreateAugmenter((3,224,224),resize=224,rand_crop=True,rand_mirror=True,mean=True))
```
#### 示例：
```
>>> data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 224, 224), label_width=1,
                                   path_imglist='data/custom.lst')
>>> data_iter.reset()
>>> for data in data_iter:
...     d = data.data[0]
...     print(d.shape)
>>> # we can apply lots of augmentations as well
>>> data_iter = mx.image.ImageIter(4, (3, 224, 224), path_imglist='data/custom.lst',
                                   rand_crop=True, rand_resize=True, rand_mirror=True, mean=True,
                                   brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1,
                                   pca_noise=0.1, rand_gray=0.05)
>>> data = data_iter.next()
>>> # specify augmenters manually is also supported
>>> data_iter = mx.image.ImageIter(32, (3, 224, 224), path_rec='data/caltech.rec',
                                   path_imgidx='data/caltech.idx', shuffle=True,
                                   aug_list=[mx.image.HorizontalFlipAug(0.5),
                                   mx.image.ColorJitterAug(0.1, 0.1, 0.1)])
```
#### 函数：
* reset(): 将图片迭代器重置为数据起点。<br>
* hard_reset(): 将图片迭代器重置为数据起点，并不管 roll over 的数据。<br>
* next_sample(): 读取下一个sample。<br>
* next(): 读取下一个batch。<br>
* check_data_shape(data_shape): 检查输入的数据形状是否有效。<br>
* check_valid_image(data): 检查输入数据是否有效。<br>
* imdecode(s): 将图片数据解码为`NDArray`格式。详见`mx.image.imdecode`。<br>
* read_image(fname): 读取一张图片。<br>
* augmentation_transform(data): 对图片使用特定增强方法。<br>
* postprocess_data(datum): 将图像加载到批处理中之前的最后一个后处理步骤。


### 2. class mx.image.ImageDetIter
```
class mxnet.image.ImageDetIter(batch_size, data_shape, path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None, shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, data_name='data', label_name='label', last_batch_handle='pad', **kwargs)
```
#### 参数：
`batch_size` int型，每个batch的图片数量。<br>
`data_shape` (channels, height, width)格式的元组，目前仅支持RGB图像。<br>
`label_width` int型，默认为1。每张图片的label数量。<br>
`path_imgrec` str型，.rec文件路径。.rec文件可由im2rec.py生成。<br>
`path_imglist` str型，.lst文件路径。.lst文件可由im2rec.py生成。<br>
`imglist` list型，图片和label的列表。每个元素为一个列表 [imagelabel：float or list of loat, imagepath]。<br>
`path_root` str型，存放图片的文件夹路径。<br>
`path_imgidx` str型，存放图片索引文件的路径。当使用.rec文件进行partition和shuffle时需要。<br>
`shuffle` bool型，是否需要在每次迭代开始前进行shuffle。<br>
`part_index` int型，partition索引。<br>
`num_parts` int型，partition的数量。<br>
`data_name` str型，数据名称。<br>
`label_name` str型，label名称。<br>
`dtype` label数据类型，默认为float32。可选参数有 int32, int64, float64。<br>
`last_batch_handle` str型，处理最后一个batch的方式，默认为'pad'。可选参数有 'pad'：则最后一个batch数据由最开始的数据来填充；'discard'：则最后一个batch的数据被丢弃；'roll_over'：将最后一个batch的数据放到下一次迭代。<br>
`aug_list` 用于进行图片增强操作，详见三. 图片增强操作。
#### 示例：
```
>>> data_iter = mx.image.ImageDetIter(batch_size=4, data_shape=(3, 224, 224),
                                      path_imglist='data/train.lst')
>>> data_iter.reset()
>>> for data in data_iter:
...     d = data.data[0]
...     l = data.label[0]
...     print(d.shape)
...     print(l.shape)
```
#### 函数：
* reshape(data_shape=None,label_shape=None): 更改图片迭代器中的图片和label形状。
* next(): 读取下一个batch。
* augmentation_transform(data, label): 对图片和label使用特定增强方法。
* check_label_shape(label_shape): 检查label的形状是否有效。
* draw_next(color=None, thickness=2, mean=None, std=None, clip=True, waitKey=None, window_name='draw_next', id2labels=None): 绘制下一张图片及bounding box。
```
>>> # use draw_next to get images with bounding boxes drawn
>>> iterator = mx.image.ImageDetIter(1, (3, 600, 600), path_imgrec='train.rec')
>>> for image in iterator.draw_next(waitKey=None):
...     # display image
>>> # or let draw_next display using cv2 module
>>> for image in iterator.draw_next(waitKey=0, window_name='disp'):
...     pass
```
* sync_label_shape(it, verbose=False): 将标签形状与输入迭代器同步。当train/validation迭代器具有不同的标签填充时，这很有用。
```
>>> train_iter = mx.image.ImageDetIter(32, (3, 300, 300), path_imgrec='train.rec')
>>> val_iter = mx.image.ImageDetIter(32, (3, 300, 300), path.imgrec='val.rec')
>>> train_iter.label_shape
(30, 6)
>>> val_iter.label_shape
(25, 6)
>>> val_iter = train_iter.sync_label_shape(val_iter, verbose=False)
>>> train_iter.label_shape
(30, 6)
>>> val_iter.label_shape
(30, 6)
```

-----------------


## 三. 图像增强操作
### 1. mx.image.CreateAugmenter
#### 函数：
```
mx.image.CreateAugmenter(
            data_shape,
            resize=0,
            rand_crop=False,
            rand_resize=False,
            rand_mirror=False,
            mean=None,
            std=None,
            brightness=0,
            contrast=0,
            saturation=0,
            hue=0,
            pca_noise=0,
            rand_gray=0,
            inter_method=2
)
```
#### 功能：
创建一个图像增强操作列表。
#### 参数：
`data_shape` int型的元组，为输出数据的形状。<br>
`resize` int型，设置较短边的resize尺寸，则较长边按照比例相应变换。<br>
`rand_crop` bool型，是否采用`mx.image.random_crop`，否则采用`center_crop`。<br>
`rand_resize` bool型，是否采用`random_size_crop`。<br>
`rand_gray` float型，取值范围[0,1]。转换为灰度图的概率。<br>
`rand_mirror` bool型，是否以0.5的概率对图片进行水平翻转。<br>
`mean` np.ndarry型或者None，RGB通道的像素平均值。<br>
`std` np.ndarry型或者None，RGB通道的像素值标准差。<br>
`brightness` float型，亮度范围。<br>
`contrast` float型，对比度范围。<br>
`saturation` float型，饱和度范围。<br>
`hue` float型，色度范围。<br>
`pca_noise` float型，PCA噪声水平。<br>
`inter_method` int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。
#### 示例：
```
>>> # An example of creating multiple augmenters
>>> augs = mx.image.CreateAugmenter(data_shape=(3, 300, 300), rand_mirror=True,
...    mean=True, brightness=0.125, contrast=0.125, rand_gray=0.05,
...    saturation=0.125, pca_noise=0.05, inter_method=10)
>>> # dump the details
>>> for aug in augs:
...    aug.dumps()
```

**`mx.image.CreateAugmenter` 创建的列表元素为封装好的图像增强操作类。<br>**
**也可自定义一个列表存储需要的图像增强操作类，以下列出所有的图像处理操作类。**

### (1) class mx.image.Augmenter(**kwargs)
#### 说明：
图像增强操作基类。
#### 函数：
* dumps(): 返回描述图像增强操作的string。

### (2) class mx.image.ResizeAug(size, interp=2)
#### 说明：
设置较短边以进行尺寸变化的增强方法。
#### 参数：
`size` int型，设置较短边的长度。<br>
`interp` int型，插值方法，默认为2。详见`mx.image.resize_short`。

### (3) class mx.image.ForceResizeAug(size, interp=2)
#### 说明：
对图片进行resize的增强方法。
#### 参数：
`size` (int,int)型的元组，为resize后的图片尺寸 (width, height)。<br>
`interp` int型，插值方法，默认为2。详见`mx.image.imresize`。

### (4) class mx.image.RandomCropAug(size, interp=2)
#### 说明：
对图片进行随机裁剪的增强方法。
#### 参数：
`size` int型，设置较短边的长度。<br>
`interp` int型，插值方法，默认为2。详见`mx.image.random_crop`。

### (5) class mx.image.RandomSizeCropAug(size, area, ratio, interp=2, **kwargs)
#### 说明：
对图片进行随机裁剪，之后再以随机面积和随机宽高比进行resize的增强方法。
#### 参数：
`size` (int,int)型的元组，为resize后的图片尺寸 (width, height)。<br>
`area` (0,1]之间的float数或（float,float)型的元组。表示所使用的面积上下限，若为float型，则上限为1.0。<br>
`ratio` (float, float)型的元组，表示宽高比的上下限。<br>
`interp` int型，插值方法，默认为2。详见`mx.image.random_size_crop`。

### (6) class mx.image.CenterCropAug(size, interp=2)
#### 说明：
对图片进行中心裁剪的增强方法。
#### 参数：
`size` (int,int)型的元组，裁剪区域的尺寸 (width, height)。<br>
`interp` int型，插值方法，默认为2。详见`mx.image.center_crop`。

### (7) class.mx.image.BrightnessJitterAug(brightness)
#### 说明：
对图片进行亮度变化的增强方法。
#### 参数：
`brightness` float型，取值范围 [0,1]。亮度变化范围。

### (8) class.mx.image.ContrastJitterAug(contrast)
#### 说明：
对图片进行对比度变化的增强方法。
#### 参数：
`contrast` float型，取值范围 [0,1]。对比度变化范围。

### (9) class.mx.image.SaturationJitterAug(saturation)
#### 说明：
对图片进行饱和度变化的增强方法。
#### 参数：
`saturation` float型，取值范围 [0,1]。饱和度变化范围。

### (10) class.mx.image.HueJitterAug(hue)
#### 说明：
对图片进行色度变化的增强方法。
#### 参数：
`hue` float型，取值范围 [0,1]。色度变化范围。

### (11) class.mx.image.ColorJitterAug(brightness, contrast, saturation)
#### 说明：
对图片进行亮度，对比度变化和饱和度的增强方法。
#### 参数：
`brightness` float型，取值范围 [0,1]。亮度变化范围。<br>
`contrast` float型，取值范围 [0,1]。对比度变化范围。<br>
`saturation` float型，取值范围 [0,1]。饱和度变化范围。

### (12) class.mx.image.LightingAug(alphastd, eigval, eigvec)
#### 说明：
对图片增加PCA噪声的增强方法。
#### 参数：
`alphastd` float型，噪声水平。<br>
`eigval` 3x1的np.array，特征值。<br>
`eigvec` 3x1的np.array，特征向量。

### (13) class mx.image.ColorNormalizeAug(mean, std)
#### 说明：
对图片像素数据进行标准化。
#### 参数：
`mean` `NDArray`型，RGB通道的像素平均值。<br>
`std` `NDArray`型，RGB通道的像素值标准差。<br>

### (14) class mx.image.RandomGrayAug(p)
#### 说明：
以一定概率将图片转为灰度图。
#### 参数：
`p` 转为灰度图的概率。

### (15) class mx.image.HorizontalFlipAug(p)
#### 说明：
以一定概率将图片进行水平翻转。
#### 参数：
`p` 水平翻转的概率。

### (16) class mx.image.CastAug(tpy='float32')
#### 说明：
将图片数据转为`float32`型。

### (17) class mx.image.SequentialAug(ts)
#### 说明：
按顺序将图片增强操作存放在列表里。

### (18) class mx.image.RandomOrderAug(ts)
#### 说明：
按随机顺序将图片增强操作存放在列表里。

### 2. mx.image.CreateDetAugmenter
#### 函数：
```
image.CreateDetAugmenter(data_shape, resize=0, rand_crop=0, rand_pad=0, rand_gray=0, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0, saturation=0, pca_noise=0, hue=0, inter_method=2, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 3.0), min_eject_coverage=0.3, max_attempts=50, pad_val=(127, 127, 127))
```
#### 功能：
创建用于object detection任务的图像增强操作列表。
#### 参数：
`data_shape` int型的元组，为输出数据的形状。<br>
`resize` int型，若resize>0，则为设置较短边的边长。<br>
`rand_crop` float型，取值范围[0,1]，进行随机裁剪的概率。<br>
`rand_pad` float型，取值范围[0,1]，进行随机填充的概率。<br>
`rand_gray` float型，取值范围[0,1]，将图片转为灰度图的概率。<br>
`rand_mirror` bool型，是否以0.5的概率对图片进行水平翻转。<br>
`mean` np.ndarry型或者None，RGB通道的像素平均值。<br>
`std` np.ndarry型或者None，RGB通道的像素值标准差。<br>
`brightness` float型，亮度范围。<br>
`contrast` float型，对比度范围。<br>
`saturation` float型，饱和度范围。<br>
`hue` float型，色度范围。<br>
`pca_noise` float型，PCA噪声水平。<br>
`inter_method` int型，默认为2。可选参数有 0：最近邻插值； 1：双线性插值；2：面积插值；3：双三次插值；4：Lanczos插值；9：三次插值；10：随机选择一种插值方法。<br>
`min_object_covered` float型，裁剪部分至少要任意bounding box百分比为此的部分。若为0，则裁剪部分不需与任意bounding box重叠。<br>
`min_eject_coverage` float型。<br>
`aspect_ratio_range` float型的元组，aspect ratio = width / height，裁剪部分的宽高比范围。<br>
`area_range` float型的元组，裁剪部分占原图尺寸的面积范围。<br>
`max_attempts` int型，尝试生成指定裁剪/填充图像的次数，若失败则返回原图。<br>
`pad_val` float型，填充的图像像素值。
#### 示例：
```
>>> # An example of creating multiple augmenters
>>> augs = mx.image.CreateDetAugmenter(data_shape=(3, 300, 300), rand_crop=0.5,
...    rand_pad=0.5, rand_mirror=True, mean=True, brightness=0.125, contrast=0.125,
...    saturation=0.125, pca_noise=0.05, inter_method=10, min_object_covered=[0.3, 0.5, 0.9],
...    area_range=(0.3, 3.0))
>>> # dump the details
>>> for aug in augs:
...    aug.dumps()
```

**object detection任务中所有的图像增强操作类如下**

### (1) class mx.image.DetAugmenter(**kwargs)
#### 说明：
图像增强操作基类。
#### 函数：
* dumps(): 返回描述图像增强操作的string。

### (2) class mx.image.DetBorrowAug(augmenter)
#### 说明：
使用三.1.中的标准图像增强方法类。

### (3) class mx.image.RandomSelectAug(aug_list, skip_prob=0)
#### 说明：
从aug_list中选择一种图像增强方法，若skip_prob=0，则有概率不进行图像增强。
#### 参数：
`aug_list` 图像增强方法列表。<br>
`skip_prob` float型，不进行图像增强直接返回原图的概率。

### (4) class mx.image.DetHorizontalFlipAug(p)
#### 说明：
随机进行图片水平翻转。

### (5) class mx.image.DetRandomCropAug(min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), min_eject_coverage=0.3, max_attempts=50)
#### 说明：
随机进行图片裁剪。
#### 参数：
`min_object_covered` float型，裁剪部分至少要任意bounding box百分比为此的部分。若为0，则裁剪部分不需与任意bounding box重叠。<br>
`min_eject_coverage` float型。<br>
`aspect_ratio_range` float型的元组，aspect ratio = width / height，裁剪部分的宽高比范围。<br>
`area_range` float型的元组，裁剪部分占原图尺寸的面积范围。<br>
`max_attempts` int型，尝试生成指定裁剪/填充图像的次数，若失败则返回原图。<br>

### (6) class mx.image.DetRandomPadAug(aspect_ratio_range=(0.75, 1.33), area_range=(1.0, 3.0), max_attempts=50, pad_val=(128, 128, 128))
#### 说明：
随机进行图片填充。
#### 参数：
`aspect_ratio_range` float型的元组，aspect ratio = width / height，裁剪部分的宽高比范围。<br>
`area_range` float型的元组，裁剪部分占原图尺寸的面积范围。<br>
`max_attempts` int型，尝试生成指定裁剪/填充图像的次数，若失败则返回原图。<br>
`pad_val` float型，填充的图像像素值。
