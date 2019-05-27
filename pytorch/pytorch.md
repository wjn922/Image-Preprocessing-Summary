## pytorch
pytoch中使用`torchvision.transforms`对`PIL.Image`及`torch.Tensor`进行图像预处理
```
from torchvison import transforms
```

### 1.对PIL.Image做变换

#### transforms.Compose(transforms)
将多个图像预处理方式`transforms`组合起来使用
```
transforms = transforms.Compose([
		transforms.CenterCrop(10),
		transforms.ToTensor()
		])
```

#### transforms.Resize(size)
将图片裁剪成给定的size大小。<br>
`size`参数可以为一个元组:(height, width)，也可以为一个整数：此时切出来的图片形状为正方形。

#### transforms.CenterCrop(size)
将给定的`PIL.Image`进行中心切割，得到给定的size。<br>
`size`参数可以为一个元组，也可以为一个整数。

#### transforms.RandomCrop(size,padding=0)
图片切割中心点的位置随机选取。<br>
`size`参数可以为一个元祖，也可以为一个整数。

#### transforms.RandomHorizontalFlip(prob)
对`PIL.Image`以概率prob进行随机水平翻转，默认概率为0.5。

#### transforms.RandomVerticalFlip(prob)
对`PIL.Image`以概率prob进行随机竖直翻转，默认概率为0.5。

#### transforms.RandomSizedCrop(size, interpolation=2)
