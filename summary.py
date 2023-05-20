#-----------------------------------------------#
# This part of the code is used to see network parameters
#-----------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50

if __name__ == "__main__":
    input_shape = [512, 512]
    num_classes = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNet_Resnet50().to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    #------------------------------------------------- -------#
    # flops * 2 is because the profile does not use convolution as two operations
    # Some papers use convolution to calculate multiplication and addition operations. multiply by 2
    # Some papers only consider the number of operations of multiplication, ignoring addition. Do not multiply by 2 at this time
    # This code chooses to multiply by 2, refer to YOLOX.
    #------------------------------------------------- -------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))