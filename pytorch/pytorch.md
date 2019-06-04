# pytorch
pytoch中使用`torchvision.transforms`对`PIL.Image`及`torch.Tensor`进行图像处理
```
from torchvison import transforms
```
------------

**pytorch的图片预处理方法可归结为四类：**
* 裁剪 -- Crop
* 翻转和旋转 -- Flip and Rotation
* 图像变换 
* transforms操作
--------

### 一. 裁剪 -- Crop

#### 1. 中心裁剪：transforms.CenterCrop(size)
**函数：** transforms.CenterCrop(size)<br>
**功能：** 对`PIL.Image`进行中心裁剪，得到给定size的图片。<br>
**参数：** `size`参数若为tuple，格式为(h,w)；若为int，裁剪得到的图片大小为(size,size)

#### 2. 随机裁剪：transforms.RandomCrop(size, padding=0)
**函数：** transforms.RandomCrop(size,padding=None,pad_if_needed=False,fill=0,padding_mode='constant')<br>
**功能：** 图片裁剪中心点的位置随机选取，将`PIL.Image`裁剪为给定size的图片。<br>
**参数：** `size`参数若为tuple，格式为(h,w)；若为int，裁剪得到的图片大小为(size,size)<br>
&emsp; &emsp; &thinsp; `padding`参数设置对图片填充多少个pixel。<br>
&emsp; &emsp; &thinsp; `fill`参数设置填充的值（仅当padding_mode='constant'时有用）。若为int，各通道均填充该值；若为长度为3的tuple，表示RGB通道需要填充的值。<br>
&emsp; &emsp; &thinsp; `padding_mode`参数用于设置填充模式。函数提供了4种模式：(1)constant; (2)edge; (3)reflect; (4)symmetric。

#### 3. 随机长宽比裁剪：transforms.RandomSizedCrop(size, interpolation=2)
**函数：** transforms.RandomSizedCrop(size, scale=(0.08,1.0), ratio=(0.75,1.33333), interpolation=2)<br>
**功能：** 采用随机大小，随机长宽比裁剪`PIL.Image`，再将图片resize到给定size大小。<br>
**参数：** `size`设置裁剪后得到的图片大小。<br>
&emsp; &emsp; &thinsp; `scale`设置随机裁剪的大小区间。如scale=(0.08,1.0)，表示随机裁剪出的图片大小会在原图的0.08-1倍之间。<br>
&emsp; &emsp; &thinsp; `ratio`设置随机裁剪的长宽比。<br>
&emsp; &emsp; &thinsp; `interpolation`设置插值的方法。

#### 4. 上下左右中心裁剪：transforms.FiveCrop(size)
**函数：** transforms.FiveCrop(size)<br>
**功能：** 对`PIL.Image`进行上下左右及中心裁剪，获得5张图片，返回一个4D-tensor。<br>
**参数：** `size`参数若为tuple，格式为(h,w)；若为int，裁剪得到的图片大小为(size,size)

#### 5. 上下左右中心裁剪后翻转： transforms.TenCrop(size)
**函数：** transforms.FiveCrop(size, vertical_flip=False)<br>
**功能：** 对`PIL.Image`进行上下左右及中心裁剪，然后全部翻转（水平翻转或者竖直翻转），获得5张图片，返回一个4D-tensor。<br>
**参数：** `size`参数若为tuple，格式为(h,w)；若为int，裁剪得到的图片大小为(size,size)<br>
&emsp; &emsp; &thinsp; `vertical_flip`设置是否垂直翻转，默认为False。


### 二. 翻转和旋转 -- Flip and Rotation

####　1. 水平翻转：transforms.RandomHorizontalFlip(p=0.5)
**函数：** transforms.RandomHorizontalFlip(p=0.5)<br>
**功能：** 以概率p对`PIL.Image`进行随机水平翻转，默认概率为0.5。<br>
**参数：** `p`设置翻转概率值。

####　2. 竖直翻转：transforms.RandomVerticalFlip(p=0.5)
**函数：** transforms.RandomVerticalFlip(p=0.5)<br>
**功能：** 以概率p对`PIL.Image`进行随机竖直翻转，默认概率为0.5。<br>
**参数：** `p`设置翻转概率值。

#### 3. 随机旋转：transforms.RandomRotation(degrees)
**函数：** transforms.RandomRotation(degrees, resample=False, expand=False, center=None)<br>
**功能：** 依据degrees对`PIL.Image`随机旋转一定角度。<br>
**参数：** `degrees`设置旋转角度区间。若为int或者float,表示在(-degrees,+degrees)之间随机旋转；若为tuple，如(30,60)，表示在(30,60)之间随机旋转。<br>
&emsp; &emsp; &thinsp; `resample`设置重采样方法。可选方式有: PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.bicubic, 默认为PIL.Image.NEAREST。<br>
&emsp; &emsp; &thinsp; `expand`bool类型，设置输出图片大小。若为True, 则输出图片大小会比原图尺寸大以覆盖旋转后的图片；若为False,　则输出图片大小与原图一致，默认为False。<br>
&emsp; &emsp; &thinsp; `center`设置旋转中心。默认旋转中心为左上角。


### 三. 图像变换 

#### 1. resize：transforms.Resize(size)
**函数：** transforms.Resize(size, interpolation=2)<br>
**功能：** 重置图片`PIL.Image`分辨率大小。<br>
**参数：** `size`设置输出图片分辨率大小。参数若为tuple，格式为(h,w)。<br>
&emsp; &emsp; &thinsp; `interpolation`设置插值的方法。

#### 2. 填充：transforms.Pad(padding)
**函数：** transforms.Pad(padding, fill=0, padding_mode='constant')<br>
**功能：** 对`PIL.Image`图像进行填充。<br>
**参数：** `padding`参数设置对图片填充多少个pixel。<br>
&emsp; &emsp; &thinsp; `fill`参数设置填充的值（仅当padding_mode='constant'时有用）。若为int，各通道均填充该值；若为长度为3的tuple，表示RGB通道需要填充的值。<br>
&emsp; &emsp; &thinsp; `padding_mode`参数用于设置填充模式。函数提供了4种模式：(1)constant; (2)edge; (3)reflect; (4)symmetric。

#### 3. 修改亮度、对比度、饱和度和色度：transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
**函数：** transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)<br>
**功能：** 对`PIL.Image`图像修改亮度、对比度、饱和度和色度。

#### 4. 仿射变换：transforms.RandomAffine(degrees)
**函数：** transforms.RandomAffine(degrees, translate=None, scale=None, shear=NOne, resample=False, fillcolor=0)<br>
**功能：** 对`PIL.Image`图像进行仿射变换。<br>

#### 5. 转为灰度图：transforms.RandomGrayscale(p=0.1)
**函数：** transforms.RandomGrayscale(p=0.1)<br>
**功能：** 以概率p将`PIL.Image`转换为灰度图。

#### 6. 转为PIL.Image：transforms.ToPILImage()
**函数：** transforms.ToPILImage()<br>
**功能：** 将shape为(c,h,w)的`torch.FloatTensor`或shape为(h,w,c)的`numpy.ndarray`转换为`PIL.Image`。

#### 7. 转为tensor：transforms.ToTensor()
**函数：** transforms.ToTensor()<br>
**功能：** 将取值范围为[0,255]的`PIL.Image`或shape为(h,w,c)的`numpy.ndarray`转换为shape为(c,h,w)的`torch.FloatTensor`，且归一化至[0,1]。

#### 8. 标准化：transforms.Normalize(mean, std)
**函数：** transforms.NOrmalize(mean, std)<br>
**功能：** 对数据`torch.Tensor`按通道进行标准化，即对数据先减均值，再除以标准差。<br>
**参数：** `mean`设置均值，`std`设置标准差，格式为三元组　(R,G,B)。注意要转为`torch.FloatTensor`后使用。

#### 9. 线性变换：transforms.LinearTransformatioin()
**函数：** transforms.LinearTransformatioin(transformation_matrix)<br>
**功能：** 对`torch.Tensor`做线性变换。<br>
**参数：** `transformation_matrix`tensor类型，维数为[DxD],D=CxHxW

#### 10. 使用自定义函数：transforms.Lambda


### 四. transforms操作

#### 1. transforms.Compose(transform)
**函数：** transforms.Compose(transforms)<br>
**功能：** 将多个图像预处理方式`transforms`组合起来使用<br>

```
transforms = transforms.Compose([
		transforms.CenterCrop(10),
		transforms.ToTensor()
		])
```

#### 2. transforms.RandomChoice(transforms)
**函数：** transforms.RandomChoice(transforms)<br>
**功能：** 从给定的一系列transforms中选一个进行操作。

#### 3. transforms.RandomApply(transforms, p=0.5)
**函数：** transforms.RandomApply(transforms, p=0.5)<br>
**功能：** 给一个`transforms`加上概率，以一定概率执行该操作。

#### 2. transforms.RandomOrder(transforms)
**函数：** transforms.RandomOrder(transforms)<br>
**功能：** 将`transforms`中的操作顺序打乱。


































