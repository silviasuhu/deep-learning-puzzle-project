
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2


###################################################
################ BLOCKS ###########################
###################################################

# This is a variant of the ResNet basic block for group-equivariant convolutions 
# in the P4 group (0º, 90º, 180º, 270º)
# Main idea is that rotating the input by n*90º produces a predictable rotation
# in the feature space, which will be useful to to learn for puzzles with rotations
class BasicBlock(nn.Module):
    expansion = 1 # Output channels are expanded by x1 (no expansion)
    def __init__(self, 
                 in_planes, # Input channels
                 planes, # Output channels
                 stride=1): #Stride
        super(BasicBlock, self).__init__() 
        # P4ConvP4 is a specific type of Group Equivariant Convolution (G-Conv) used in CNNs
        #  to handle data with rotational symmetry. 
        #  It operates on the p4 group, which consists of 90-degree rotations and translation
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes) # Becase we have a "rotation" dimension, we need 3d batch norm
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential() # The default shortcut (for residual) is the identity (no modification)

        # If spatial or channel dimension changes, we do a projection shortcut (residual)
        # 1x1 convolution + BatchNorm to match dimensions
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Conv --> BN --> Relu
        out = self.bn2(self.conv2(out)) # Conv --> BN
        out += self.shortcut(x) # Add residual (either the identity (nothing) or the computed one)
        out = F.relu(out) # Relu
        return out

# Basic Bottleneck module
class Bottleneck(nn.Module):
    expansion = 4 # Output channels are expanded by x4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # Same as above, define the P4ConvP4 convolution and BatchNorm layers
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = P4ConvP4(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        # Same as above, the (optional) residual 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Conv --> BN --> Relu
        out = F.relu(self.bn2(self.conv2(out))) # Conv --> BN --> Relu
        out = self.bn3(self.conv3(out)) # Conv --> BN
        out += self.shortcut(x) # Add residual
        out = F.relu(out) # Relu
        return out

###################################################
################ MODELS ###########################
###################################################

# Actual Resnet Equivariant variant model
class ResNet(nn.Module):
    def __init__(self, 
                 block, # Which block to ux
                 num_blocks, #  Number of blocks to build in each of the 4 "layers" --> [n1,n2,n3,n4]
                 num_classes=10 # Actually just unused in this implementation. In normal resnet, 
                                # used for classification, it's the number of output classes
                 ):
        super(ResNet, self).__init__()
        self.in_planes = 32 # Initialize channel width

        ###### IMPORTANT, EQUIVARIANCE EXPLANATION #########
        # First layer feeds the image (Z2 --> input channels = 3 channels -RGB-) to the P4 feature space  (32=4*8)
        # This is the key of the "equivariant" part of this Resnet
        # Z2 is the group a normal image is --> ie: equivariant to translation only in the 2D grid
        # P4 is a group that is equivariant to rotations in 4 discrete positions (0º, 90º, 180º, 270º) + translation 
        # So, OVERALL:
        # 1.- We rotate the patches during dataset creation in a discrete way
        # 2.- We pass them through this equivariant ResNet
        # 3.- This produces features that change PREDICTABLY with the rotation
        # 4.- This way, the feature reflect "how the patch change when X rotation is applied"
        # 5.- When feeding this features to the GNN (and the ground truth rotation as target), the GNN learns HOW
        # features reflect rotation for the patches
        # 6.- Finally, the GNN uses the learned information to predict rotations that need to be applied to each patch
        # when no ground truth (known rotation) is provided
        self.conv1 = P4ConvZ2(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes) # 3D Normalization (HxWxRotation)

        # 4 layers, each with the given number of residual or bottleneck blocks
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1) # First layer, with n1 blocks
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2) # Second layer, with n2 blocks
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) # Third layer, with n3 blocks
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2) # Fourth layer, with n4 blocks
        #self.linear1 = nn.Linear(256*4*block.expansion, num_classes)
        self.linear1 = nn.Linear(128*4*block.expansion*32, 544)  # FC layer --> Projection head for out3 features
        self.linear2 = nn.Linear(256*4*block.expansion*8, 544) # FC layer --> Projection head for out4 features

    # Function to create each of the above layers
    # Gets the type and nubmer of blocks, stride, and planes (input channels) and stacks them updating the num channels (planes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # Only the first block might have a diff stride (s=2), the rest are s=1
        layers = [] # Initialize list of blocks
        for stride in strides: # iterative styacking
            layers.append(block(self.in_planes, planes, stride)) # Append block to "layers" with the charact of the block
            self.in_planes = planes * block.expansion # Update planes/channel width
        return nn.Sequential(*layers) # Return a nn.Sequential with the stack of blocks

    # Main forward pass
    def forward(self, x):
        self.conv1(x) # This is unassigned, seems an artifact

        out = F.relu(self.bn1(self.conv1(x))) # Conv1 (converts Z2 image to P4) --> BN --> Relu
        out1 = self.layer1(out) # First layer of blocks
        out2 = self.layer2(out1) # Second layer of blocks
        out3 = self.layer3(out2)    # Third layer of blocks
        out4 = self.layer4(out3) # Forth layer of blocks
        
        #outs = out4.size()
        #out = out4.view(outs[0], outs[1]*outs[2], outs[3], outs[4])
        #out = F.avg_pool2d(out, 4)
        #print(out.size())
        #out = out.view(out.size(0), -1)

        # Now we get the flatten vector feature of out3 and out4
        # The backbone models like Efficient_GAT actually concatenate out3+out4 and use them both as combined feature
        # I guess the idea is to get 2 different levels of encoded features per each patch
        out3 = self.linear1(out3.view(out3.size(0), -1)) # Flatten out3 and project to 544
        out4 = self.linear2(out4.view(out4.size(0), -1)) # Flatten out4 and project to 544
        

        # output layer
        #out = self.linear(out)
        return [out1, out2, out3, out4] # Return intermediate features (out1, out2) and projected head vectors (oput3, out4)


###################
# These are honestly very silly ways to call predetermined Resnet Equivariant version that they might have tried.
# We can make our own

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

# Just a test
def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())