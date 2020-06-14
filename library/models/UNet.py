from imports.imports_eva import *

class UNet (nn.Module):

    #Contracting Block
    def contractBlock (self, in_channels, out_channels, kernel_size = 3):
        block = nn.Sequential (
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d (num_features = out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d (num_features = out_channels),
            nn.ReLU()
        )
        return block

    def expandBlock (self, in_channels, mid_channels, out_channels, kernel_size = 3):
        block = nn.Sequential (
            nn.Conv2d (in_channels = in_channels, out_channels = mid_channels, kernel_size = kernel_size, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (mid_channels),
            nn.ReLU(),

            nn.Conv2d (in_channels = mid_channels, out_channels = mid_channels, kernel_size = kernel_size, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (mid_channels),
            nn.ReLU(),

            #Up Sampling
            nn.ConvTranspose2d (in_channels = mid_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        )
        return block

    def finalBlock (self, in_channels, mid_channels, out_channels, kernel_size = 3, kernel_size_final = 1):
        block = nn.Sequential (
            nn.Conv2d (in_channels = in_channels, out_channels = mid_channels, kernel_size = kernel_size, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (mid_channels),
            nn.ReLU(),

            nn.Conv2d (in_channels = mid_channels, out_channels = mid_channels, kernel_size = kernel_size, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (mid_channels),
            nn.ReLU(),

            nn.Conv2d (in_channels = mid_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU()
        )
        return block

    def __init__ (self, in_channels, out_channels):
        super (UNet, self).__init__()

        #Encode
        self.convblock1 = self.contractBlock (in_channels = in_channels, out_channels = 64)
        self.pool1 = nn.MaxPool2d (kernel_size = 2, stride = 2)

        self.convblock2 = self.contractBlock (in_channels = 64, out_channels = 128)
        self.pool2 = nn.MaxPool2d (kernel_size = 2, stride = 2)

        self.convblock3 = self.contractBlock (in_channels = 128, out_channels = 256)
        self.pool3 = nn.MaxPool2d (kernel_size = 2, stride = 2)

        self.convblock4 = self.contractBlock (in_channels = 256, out_channels = 512)
        self.pool4 = nn.MaxPool2d (kernel_size = 2, stride = 2)

        #Base
        self.base = nn.Sequential (
            nn.Conv2d (in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (num_features= 1024),
            nn.ReLU(),

            nn.Conv2d (in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d (num_features = 1024),
            nn.ReLU(),

            nn.ConvTranspose2d (in_channels = 1024, out_channels = 512, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        )

        # Decode
        self.convblock5 = self.expandBlock (in_channels = 1024, mid_channels = 512, out_channels = 256)

        self.convblock6 = self.expandBlock (in_channels = 512, mid_channels = 256, out_channels = 128)

        self.convblock7 = self.expandBlock (in_channels = 256, mid_channels = 128, out_channels = 64)

        self.convblock_final = self.finalBlock (in_channels = 128, mid_channels = 64, out_channels = out_channels)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward (self, x):

        # Encode
        conv_encode_1 = self.convblock1 (x)
        conv_encode_pool_1 = self.pool1 (conv_encode_1)

        conv_encode_2 = self.convblock2 (conv_encode_pool_1)
        conv_encode_pool_2 = self.pool2 (conv_encode_2)

        conv_encode_3 = self.convblock3 (conv_encode_pool_2)
        conv_encode_pool_3 = self.pool3 (conv_encode_3)

        conv_encode_4 = self.convblock4 (conv_encode_pool_3)
        conv_encode_pool_4 = self.pool4 (conv_encode_4)

        # Base
        conv_base = self.base (conv_encode_pool_4)

        # Decode
        concat_1 = self.crop_and_concat (conv_base, conv_encode_4, crop = True)
        conv_decode_1 = self.convblock5 (concat_1)

        concat_2 = self.crop_and_concat (conv_decode_1, conv_encode_3, crop = True)
        conv_decode_2 = self.convblock6 (concat_2)

        concat_3 = self.crop_and_concat (conv_decode_2, conv_encode_2, crop = True)
        conv_decode_3 = self.convblock7 (concat_3)

        concat_3 = self.crop_and_concat (conv_decode_3, conv_encode_1, crop = True)
        conv_final_layer = self.convblock_final (concat_3)

        return conv_final_layer