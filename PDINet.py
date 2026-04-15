import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Callable, Optional
from rbf.basis import get_rbf  # If you have trouble installing rbf, you can comment out this line and load precomputed kernels from local files.



class PDI_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PDI_Layer, self).__init__()
        self.conv1 = torch.nn.Conv2d(6 * in_channels - 5, out_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # By default, generate Gaussian derivative kernels using rbf.
        kernel1 = self.make_gauss(1, 9).cuda().reshape(2, 1, 1, 9, 9)
        kernel2 = self.make_gauss(2, 9).cuda().reshape(3, 1, 1, 9, 9)
        '''
        If you have trouble installing rbf, comment out the above two lines,
        and use the following two lines to load precomputed kernels from local files:
        kernel1 = torch.load("kernel1.pt").cuda().reshape(2, 1, 1, 9, 9)
        kernel2 = torch.load("kernel2.pt").cuda().reshape(3, 1, 1, 9, 9)
        '''
        self.conv_weights = [kernel1[0], kernel1[1],
                             kernel2[0], kernel2[2], kernel2[1]]
        self.stride = stride

    def dx(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[0]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=4, stride=1, groups=c)

    def dy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[1]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=4, stride=1, groups=c)
    
    def dxx(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[2]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=4, stride=1, groups=c)

    def dyy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[3]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=4, stride=1, groups=c)
    
    def dxy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[4]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=4, stride=1, groups=c)

    def make_coord(self, kernel_size):
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        coord = torch.meshgrid([-x, x])
        coord = torch.stack([coord[1], coord[0]], -1)
        return coord.reshape(kernel_size ** 2, 2)

    def make_gauss(self, order, kernel_size):
        diff = []
        coord = self.make_coord(kernel_size)
        gauss = get_rbf('ga')
        for i in range(order + 1):
            w = gauss(coord, torch.zeros(1, 2), eps=0.99, diff=[i, order - i]).reshape(kernel_size, kernel_size)
            w = torch.tensor(w)
            diff.append(w)
        tensor = torch.stack(diff, 0)
        return tensor.to(torch.float32)

    def normalize(self, inv):
        batch = inv.shape[0]
        max_values = inv.abs().view(batch, -1).max(dim=-1)[0]
        max_values[max_values==0] = 1
        inv = inv / max_values.view(batch, 1, 1, 1)
        return inv

    def divide(self, numerator, denominator):
        denominator = torch.where(denominator == 0,
                                  torch.tensor(1.0, device=denominator.device), denominator)
        return numerator / denominator

    def set_R0(self, u):
        ux = self.dx(u)
        uy = self.dy(u)
        ux0 = ux[:,0:1,:,:]
        ux1 = ux[:,1:2,:,:]
        ux2 = ux[:,2:3,:,:]
        uy0 = uy[:,0:1,:,:]
        uy1 = uy[:,1:2,:,:]
        uy2 = uy[:,2:3,:,:]

        R0_1 = ux0 * uy1 - ux1 * uy0
        R0_2 = ux1 * uy2 - ux2 * uy1
        R0_3 = ux2 * uy0 - ux0 * uy2
        self.R0 = (R0_1 + R0_2 + R0_3) / 3

    def compute_projective_invariants(self, u):
        eps = 1
        ux = self.dx(u)
        uy = self.dy(u)
        uxx = self.dxx(u)
        uyy = self.dyy(u)
        uxy = self.dxy(u)
        uxx_uy = uxx * uy
        uxy_ux = uxy * ux
        uxy_uy = uxy * uy
        uyy_ux = uyy * ux
        shift = list(range(1, u.shape[1]))
        shift.append(0)
        
        inv2 = uxx_uy * uy - 2 * uxy_ux * uy + uyy_ux * ux
        inv3 = 2 * (uxx_uy[:,:-1,:,:] * uy[:,1:,:,:] - uxy_ux[:,:-1,:,:] * uy[:,1:,:,:] -
                    uxy_uy[:,:-1,:,:] * ux[:,1:,:,:] + uyy_ux[:,:-1,:,:] * ux[:,1:,:,:]) + \
                (uxx[:,1:,:,:] * uy[:,:-1,:,:]**2 - 2 * uxy[:,1:,:,:] * ux[:,:-1,:,:] * uy[:,:-1,:,:] + uyy[:,1:,:,:] * ux[:,:-1,:,:]**2)
        inv4 = 2 * (uxx_uy[:,1:,:,:] * uy[:,:-1,:,:] - uxy_ux[:,1:,:,:] * uy[:,:-1,:,:] -
                     uxy_uy[:,1:,:,:] * ux[:,:-1,:,:] + uyy_ux[:,1:,:,:] * ux[:,:-1,:,:]) + \
                (uxx[:,:-1,:,:] * uy[:,1:,:,:]**2 - 2 * uxy[:,:-1,:,:] * ux[:,1:,:,:] * uy[:,1:,:,:] + uyy[:,:-1,:,:] * ux[:,1:,:,:]**2)
        inv234 = torch.cat((inv2, inv3, inv4), 1)
        
        inv5 = ux[:,:-2,:,:] * uy[:,2:,:,:] - uy[:,:-2,:,:] * ux[:,2:,:,:]
        inv6 = ux[:,:-1,:,:] * uy[:,1:,:,:] - uy[:,:-1,:,:] * ux[:,1:,:,:]
        inv56 = torch.cat((inv5, inv6), 1)

        R0 = self.R0
        
        inv1 = self.normalize(u)
        inv234 = self.normalize(self.divide(inv234, eps + R0**2))
        inv56 = self.normalize(self.divide(inv56, eps + R0))

        return torch.cat((inv1, inv234, inv56), 1)

    def forward(self, x):
        x = self.compute_projective_invariants(x)
        x = self.conv1(x)
        x = F.relu(x)
        if self.stride == 2:
            x = F.avg_pool2d(x, (2, 2))
        x = self.conv2(x)
        return x



class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = PDI_Layer(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PDI_Layer(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, image: Tensor) -> Tensor:
        while x.shape[-1] < image.shape[-1]:
            image = F.avg_pool2d(image, (2, 2))
        self.conv1.set_R0(image)
        if self.conv1.stride == 2:
            image = F.avg_pool2d(image, (2, 2))
        self.conv2.set_R0(image)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class CustomSequential(nn.Sequential):
    def forward(self, x, image):
        for module in self:
            x = module(x, image)
        return x



class PDINet_ResNet50(nn.Module):

    def __init__(
        self,
        block = BasicBlock,
        layers = [2, 2, 2, 2],
        num_classes: int = 10,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(PDINet_ResNet50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.conv1 = PDI_Layer(3, self.inplanes, stride=2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return CustomSequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        self.image = x
        self.conv1.set_R0(self.image)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, self.image)
        x = self.layer2(x, self.image)
        x = self.layer3(x, self.image)
        x = self.layer4(x, self.image)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return torch.log_softmax(x, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)