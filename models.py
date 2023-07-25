import torch.nn.functional as F
from torch import nn
import torch
from collections import OrderedDict

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25))
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=2, p=1, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, out_channels, k, stride=s, padding=p, **kwargs),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded * x

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('ae1', Autoencoder(64, 64, output_padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('ae2', Autoencoder(64, 64, k=3, s=3, p=3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('ae1', Autoencoder(128, 128, s=4, output_padding=2)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('ae2', Autoencoder(128, 128, k=2, s=4, output_padding=1)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x
    
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ECACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('eca1', eca_layer(64)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('eca2', eca_layer(64)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('eca1', eca_layer(128)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('eca2', eca_layer(128)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        # print(x.shape, self.ae(x)[1].shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

class AutoencoderECACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('eca1', eca_layer(64)),
            ('ae1', Autoencoder(64, 64, output_padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('eca2', eca_layer(64)),
            ('ae2', Autoencoder(64, 64, k=3, s=3, p=3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('eca1', eca_layer(128)),
            ('ae1', Autoencoder(128, 128, s=4, output_padding=2)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('eca2', eca_layer(128)),
            ('ae2', Autoencoder(128, 128, k=2, s=4, output_padding=1)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size =7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ECA_Spatial(nn.Module):
    def __init__(self, gate_channels):
        super(ECA_Spatial, self).__init__()
        self.ChannelGate = eca_layer(gate_channels)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class ECASpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('att', ECA_Spatial(64)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('att', ECA_Spatial(64)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),

        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('att', ECA_Spatial(128)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('att', ECA_Spatial(128)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),

        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # print(x.shape, self.ae(x)[1].shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x


class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),

        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),

        ]))

        self.layer3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 256, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(256)),
            ('conv2', nn.Conv2d(256, 256, 3)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(256)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),

        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 3, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x
    
class LinearAutoencoder(nn.Module):
    def __init__(self, input_size=32 * 32, layers=[128, 64, 12, 3]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(layers[2], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], input_size),
            nn.Sigmoid(),
        )
        for layer in [*self.encoder.modules(), *self.decoder.modules()]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x * decoded
    
class LinearAutoencoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('ae1', LinearAutoencoder(32)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('ae2', LinearAutoencoder(30)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('ae1', LinearAutoencoder(15)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('ae2', LinearAutoencoder(13)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

class LinearAutoencoderECACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('eca1', eca_layer(64)),
            ('ae1', LinearAutoencoder(32)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('eca2', eca_layer(64)),
            ('ae2', LinearAutoencoder(30)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('eca1', eca_layer(128)),
            ('ae1', LinearAutoencoder(15)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('eca2', eca_layer(128)),
            ('ae2', LinearAutoencoder(13)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    
class CBAMCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
            ('att',CBAM(64)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(64)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('att',CBAM(64)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(64)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, 3, padding=1)),
            ('att',CBAM(128)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(128)),
            ('conv2', nn.Conv2d(128, 128, 3)),
            ('att',CBAM(128)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(128)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
        ]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x