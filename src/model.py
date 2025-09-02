class ViT_EfficientNet_Hybrid(nn.Module):
    def __init__(self, num_classes=51, efficientnet_model_name='efficientnet_b0', vit_model_name='vit_small_patch16_224', freeze_layer='blocks.5', dropout=0.5):
        super(ViT_EfficientNet_Hybrid, self).__init__()

        # Load EfficientNet model
        self.efficientnet = timm.create_model(efficientnet_model_name, pretrained=True)

        # Freeze layers before the specified block in EfficientNet
        self.freeze_layers(freeze_layer)

        # Extract the blocks for EfficientNet model and build the feature extractor
        self.feature_extractor = nn.Sequential(*self.get_efficientnet_blocks(freeze_layer))

        # Load pre-trained Vision Transformer (ViT)
        self.vit = create_model(vit_model_name, pretrained=True)

        # Modify ViT's classifier for the final output
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        # Optional dropout layer before the final output
        self.dropout = nn.Dropout(dropout)

    def freeze_layers(self, freeze_layer):
        """Freeze layers before the specified block"""
        freeze = False
        for name, param in self.efficientnet.named_parameters():
            if freeze:
                param.requires_grad = False
            if name.startswith(freeze_layer):
                freeze = True

    def get_efficientnet_blocks(self, freeze_layer):
        """Get EfficientNet blocks before the freeze layer"""
        blocks = list(self.efficientnet.children())  # Get the top-level layers
        # EfficientNet uses a structure where the main blocks are contained inside a 'features' layer
        features = blocks[0] if isinstance(blocks[0], nn.Sequential) else blocks[1]
        # Now, we need to extract the blocks properly from the 'features' layer
        feature_blocks = list(features.children())  # Get the individual blocks in the features part
        
        # Return all blocks before the freeze layer
        freeze_index = next((i for i, block in enumerate(feature_blocks) if str(block).startswith(freeze_layer)), len(feature_blocks))
        return feature_blocks[:freeze_index]

    def forward(self, x):
        N, D, C, H, W = x.shape  # N: batch_size, D: num_frames, C: channels, H: height, W: width

        # Step 1: Extract spatial features from each frame using EfficientNet
        frame_features = []

        for i in range(D):
            frame = x[:, i, :, :, :]  # Extract the ith frame from the batch
            efficientnet_output = self.feature_extractor(frame)  # Extract features using EfficientNet
            frame_features.append(efficientnet_output.unsqueeze(1))  # Add a dimension for time

        # Stack the frame features along the time dimension (N, D, feature_size)
        frame_features = torch.cat(frame_features, dim=1)  # (N, D, feature_size)

        # Step 2: Flatten the frames for ViT (reshape to [N*D, C, H, W])
        batch_size, num_frames, C, H, W = frame_features.shape
        frame_features = frame_features.view(batch_size * num_frames, C, H, W)  # [N*D, C, H, W]

        # Step 3: Pass the frame features through ViT
        vit_output = self.vit(frame_features)  # Shape: [N*D, num_classes]

        # Step 4: Reshape the output to match the batch and temporal dimensions
        vit_output = vit_output.view(batch_size, num_frames, -1)  # Reshape to [N, D, num_classes]

        # Step 5: Average over the temporal dimension (frame predictions) to get a final prediction
        vit_output = vit_output.mean(dim=1)  # Now we have [N, num_classes] by averaging across frames

        # Step 6: Apply dropout
        vit_output = self.dropout(vit_output)

        return vit_output