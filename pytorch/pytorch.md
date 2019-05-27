# pytorch
pytoch中使用`torchvision.transforms`对`PIL.Image`及`torch.Tensor`进行图像预处理
```
from torchvison import transforms
```

**pytorch的图片可归结为四类： 裁剪，翻转和旋转，图像变换，和对transforms操作**


#### transforms.Compose(transforms)
将多个图像预处理方式`transforms`组合起来使用
```
transforms = transforms.Compose([
		transforms.CenterCrop(10),
		transforms.ToTensor()
		])
```

### 一. 裁剪 -- Crop

#### 1. 中心裁剪： transforms.CenterCrop(size)
函数： transforms.CenterCrop(size)<br>
功能： 对`PIL.Image`进行中心裁剪，得到给定size的图片。<br>
参数： `size`参数若为tuple，格式为(h,w)；若为int，裁剪得到的图片大小为(size,size)

#### 2. 随机裁剪： transforms.RandomCrop(size,padding=0)
函数： transforms.RandomCrop(size,padding=None,pad_if_needed=False,fill=0,padding_mode='constant')<br>
功能： 图片切割中心点的位置随机选取，将`PIL.Image`裁剪为给定size的图片。<br>
参数： `size`参数若为tuple，格式为(h,w)；若为int，裁剪得到的图片大小为(size,size)
&nbsp;&nbsp;`padding`参数设置对图片填充多少个pixel。<br>
&nbsp;&nbsp;`fill`参数设置填充的值（仅当padding_mode='constant'时有用）。若为int，各通道均填充该值；若为长度为3的tuple，表示RGB通道需要填充的值。<br>
&nbsp;&nbsp;`padding_mode`参数用于设置填充模式。函数提供了4种模式：(1)constant; (2)edge; (3)reflect; (4)symmetric。

#### 3. 随机长宽比裁剪： transforms.RandomSizedCrop(size, interpolation=2)
函数： transforms.RandomSizedCrop(size, scale=(0.08,1.0), ratio=(0.75,1.33333), interpolation=2)<br>
功能： 采用随机大小，随机长宽比裁剪`PIL.Image`，再将图片resize到给定size大小。
参数： `size`设置裁剪后得到的图片大小。
&nbsp;&nbsp;`scale`设置随机裁剪的大小区间。如scale=(0.08,1.0)，表示随机裁剪出的图片大小会在原图的0.08-1倍之间。
&nbsp;&nbsp;`ratio`设置随机裁剪的长宽比。
&nbsp;&nbsp;`interpolation`设置插值的方法。


#### transforms.RandomHorizontalFlip(prob)
对`PIL.Image`以概率prob进行随机水平翻转，默认概率为0.5。

#### transforms.RandomVerticalFlip(prob)
对`PIL.Image`以概率prob进行随机竖直翻转，默认概率为0.5。

#### transforms.Resize(size)
将图片裁剪成给定的size大小。<br>
`size`参数可以为一个元组:(height, width)，也可以为一个整数：此时切出来的图片形状为正方形。




