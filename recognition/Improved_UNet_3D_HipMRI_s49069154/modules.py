"""Standard 2D, standard 3D, and improved 3D U-Net models for HipMRI."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class DoubleConv2D(nn.Module):
    """Two convolution-normalisation-activation layers for the 2D baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class UpBlock2D(nn.Module):
    """Upsample a 2D map, concatenate its encoder skip, and refine it."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.refine = DoubleConv2D(out_channels + skip_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = self.upsample(inputs)
        if inputs.shape[2:] != skip.shape[2:]:
            inputs = functional.interpolate(
                inputs,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.refine(torch.cat((skip, inputs), dim=1))


class UNet2D(nn.Module):
    """Basic four-level 2D U-Net used as the slice-wise baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**level) for level in range(5)]
        self.encoder1 = DoubleConv2D(in_channels, channels[0])
        self.encoder2 = DoubleConv2D(channels[0], channels[1])
        self.encoder3 = DoubleConv2D(channels[1], channels[2])
        self.encoder4 = DoubleConv2D(channels[2], channels[3])
        self.bottleneck = DoubleConv2D(channels[3], channels[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder4 = UpBlock2D(channels[4], channels[3], channels[3])
        self.decoder3 = UpBlock2D(channels[3], channels[2], channels[2])
        self.decoder2 = UpBlock2D(channels[2], channels[1], channels[1])
        self.decoder1 = UpBlock2D(channels[1], channels[0], channels[0])
        self.segmentation_head = nn.Conv2d(
            channels[0], num_classes, kernel_size=1
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_2d_input(inputs)
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))

        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        return self.segmentation_head(decoder1)


class DoubleConv3D(nn.Module):
    """Two convolution-normalisation-activation layers for the baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class ResidualBlock3D(nn.Module):
    """Pre-activation residual block used by the improved architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.main(inputs) + self.shortcut(inputs)


class UpBlock3D(nn.Module):
    """Upsample, concatenate an encoder skip, and refine the result."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_type: type[nn.Module],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        if block_type is ResidualBlock3D:
            self.refine = block_type(
                out_channels + skip_channels, out_channels, dropout=dropout
            )
        else:
            self.refine = block_type(out_channels + skip_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = self.upsample(inputs)
        if inputs.shape[2:] != skip.shape[2:]:
            inputs = functional.interpolate(
                inputs,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return self.refine(torch.cat((skip, inputs), dim=1))


class UNet3D(nn.Module):
    """Standard 3D U-Net baseline with four resolution levels."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**level) for level in range(5)]
        self.encoder1 = DoubleConv3D(in_channels, channels[0])
        self.encoder2 = DoubleConv3D(channels[0], channels[1])
        self.encoder3 = DoubleConv3D(channels[1], channels[2])
        self.encoder4 = DoubleConv3D(channels[2], channels[3])
        self.bottleneck = DoubleConv3D(channels[3], channels[4])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.decoder4 = UpBlock3D(
            channels[4], channels[3], channels[3], DoubleConv3D
        )
        self.decoder3 = UpBlock3D(
            channels[3], channels[2], channels[2], DoubleConv3D
        )
        self.decoder2 = UpBlock3D(
            channels[2], channels[1], channels[1], DoubleConv3D
        )
        self.decoder1 = UpBlock3D(
            channels[1], channels[0], channels[0], DoubleConv3D
        )
        self.segmentation_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_3d_input(inputs)
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))

        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        return self.segmentation_head(decoder1)


class ImprovedUNet3D(nn.Module):
    """Residual 3D U-Net with learned downsampling and deep supervision."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 16,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**level) for level in range(5)]
        self.deep_supervision = deep_supervision

        self.encoder1 = ResidualBlock3D(in_channels, channels[0])
        self.down1 = _downsample(channels[0], channels[1])
        self.encoder2 = ResidualBlock3D(channels[1], channels[1])
        self.down2 = _downsample(channels[1], channels[2])
        self.encoder3 = ResidualBlock3D(channels[2], channels[2], dropout=0.10)
        self.down3 = _downsample(channels[2], channels[3])
        self.encoder4 = ResidualBlock3D(channels[3], channels[3], dropout=0.20)
        self.down4 = _downsample(channels[3], channels[4])
        self.bottleneck = ResidualBlock3D(
            channels[4], channels[4], dropout=0.30
        )

        self.decoder4 = UpBlock3D(
            channels[4], channels[3], channels[3], ResidualBlock3D, dropout=0.20
        )
        self.decoder3 = UpBlock3D(
            channels[3], channels[2], channels[2], ResidualBlock3D, dropout=0.10
        )
        self.decoder2 = UpBlock3D(
            channels[2], channels[1], channels[1], ResidualBlock3D
        )
        self.decoder1 = UpBlock3D(
            channels[1], channels[0], channels[0], ResidualBlock3D
        )

        self.segmentation_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.auxiliary_head2 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.auxiliary_head3 = nn.Conv3d(channels[2], num_classes, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        _validate_3d_input(inputs)
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.down1(encoder1))
        encoder3 = self.encoder3(self.down2(encoder2))
        encoder4 = self.encoder4(self.down3(encoder3))
        bottleneck = self.bottleneck(self.down4(encoder4))

        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        logits = self.segmentation_head(decoder1)

        if self.training and self.deep_supervision:
            output_shape = logits.shape[2:]
            auxiliary2 = functional.interpolate(
                self.auxiliary_head2(decoder2),
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            )
            auxiliary3 = functional.interpolate(
                self.auxiliary_head3(decoder3),
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            )
            return logits, auxiliary2, auxiliary3
        return logits


def _downsample(in_channels: int, out_channels: int) -> nn.Sequential:
    """Learned stride-two downsampling used by the improved model."""
    return nn.Sequential(
        nn.InstanceNorm3d(in_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
    )


def _validate_2d_input(inputs: Tensor) -> None:
    if inputs.ndim != 4:
        raise ValueError(
            "Expected input shape (batch, channels, height, width), "
            f"received {tuple(inputs.shape)}."
        )
    if any(dimension < 16 for dimension in inputs.shape[2:]):
        raise ValueError("Every spatial input dimension must be at least 16 pixels.")


def _validate_3d_input(inputs: Tensor) -> None:
    if inputs.ndim != 5:
        raise ValueError(
            "Expected input shape (batch, channels, depth, height, width), "
            f"received {tuple(inputs.shape)}."
        )
    if any(dimension < 16 for dimension in inputs.shape[2:]):
        raise ValueError("Every spatial input dimension must be at least 16 voxels.")


def build_model(
    model_name: str = "improved",
    in_channels: int = 1,
    num_classes: int = 6,
    base_channels: int = 16,
    deep_supervision: bool = True,
) -> nn.Module:
    """Build one of the three comparison models from a stable API."""
    normalised_name = model_name.lower().replace("-", "_")
    if normalised_name in {"2d", "baseline2d", "unet2d"}:
        return UNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )
    if normalised_name in {"baseline", "unet", "unet3d"}:
        return UNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )
    if normalised_name in {"improved", "improved_unet", "improved_unet3d"}:
        return ImprovedUNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
        )
    raise ValueError(
        f"Unknown model '{model_name}'. Choose '2d', 'baseline', or 'improved'."
    )


def is_2d_model_name(model_name: str) -> bool:
    """Return whether a saved or requested model name selects the 2D baseline."""
    normalised_name = model_name.lower().replace("-", "_")
    return normalised_name in {"2d", "baseline2d", "unet2d"}


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated during training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _smoke_test(
    model_names: Sequence[str],
    input_shape: tuple[int, int, int],
    base_channels: int,
) -> None:
    for model_name in model_names:
        sample_shape = input_shape[-2:] if is_2d_model_name(model_name) else input_shape
        sample = torch.randn(1, 1, *sample_shape)
        model = build_model(model_name, base_channels=base_channels)
        model.eval()
        with torch.no_grad():
            logits = model(sample)
        print(f"\n{model_name}:")
        print(f"  parameters: {count_trainable_parameters(model):,}")
        print(f"  evaluation output: {tuple(logits.shape)}")

        if isinstance(model, ImprovedUNet3D):
            model.train()
            with torch.no_grad():
                supervised_outputs = model(sample)
            print(
                "  deep-supervision outputs:",
                [tuple(output.shape) for output in supervised_outputs],
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the HipMRI U-Net models.")
    parser.add_argument(
        "--model",
        choices=("2d", "baseline", "improved", "all"),
        default="all",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=(16, 32, 32),
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help="Synthetic input shape for a quick CPU test.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=8,
        help="Base feature width for the smoke test.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    model_names = (
        ("2d", "baseline", "improved") if args.model == "all" else (args.model,)
    )
    _smoke_test(
        model_names,
        input_shape=tuple(args.input_shape),
        base_channels=args.base_channels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

