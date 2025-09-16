
import torch
from torch import nn

from kornia import metrics


def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    reduction: str = "mean",
    padding: str = "same",
) -> torch.Tensor:
    r"""Function that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim` for details about SSIM.

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> loss = ssim_loss(input1, input2, 5)
    """
    # compute the ssim map
    ssim_map: torch.Tensor = metrics.ssim(img1, img2, window_size, max_val, eps, padding)

    # compute and reduce the loss
    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    # print(loss.shape) # torch.Size([1, 3, 567, 1008])
    # exit()
    # if mask is not None:
    #     loss = loss[mask]

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return loss

# https://github.com/ian-jihoonpark/X-Diffusion/blob/985f4e46a9f66c81e03fe7346a7aa601d4288582/ssim.py#L28
# https://github.com/W-Ted/GScream/blob/fdaf32ec4fa0dcd218bfd5312d509896239c73cf/utils/my_ssim.py#L101
class SSIMLoss(nn.Module):
    r"""Create a criterion that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim_loss` for details about SSIM.

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> criterion = SSIMLoss(5)
        >>> loss = criterion(input1, input2)
    """

    def __init__(
        self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, reduction: str = "mean", padding: str = "same"
    ) -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps: float = eps
        self.reduction: str = reduction
        self.padding: str = padding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction, self.padding)

# https://github.com/lppllppl920/EndoscopyDepthEstimation-Pytorch/blob/bc32e07ea218229b61efa0047c8ea2040c942d4c/losses.py
class ScaleInvariantLoss(nn.Module):
    def __init__(self, device, epsilon=1.0e-8, reduction="mean", lamda=0.85):
        super(ScaleInvariantLoss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().to(device)
        self.reduction = reduction
        self.lamda = lamda

    def forward(self, predict, goal):
        '''
        predict/ goal: n*w*h
        '''
        depth_ratio_map = torch.log(predict + self.epsilon) - \
                          torch.log(goal + self.epsilon)

        weighted_sum = torch.sum(torch.ones_like(predict), dim=(1, 2))
        loss_1 = torch.sum(depth_ratio_map * depth_ratio_map,
                           dim=(1, 2)) / weighted_sum
        sum_2 = torch.sum(depth_ratio_map, dim=(1, 2))
        loss_2 = (sum_2 * sum_2) / (weighted_sum * weighted_sum)
        
        if self.reduction == "mean":
            loss = torch.mean(loss_1 - self.lamda*loss_2)
        elif self.reduction == "sum":
            loss = torch.sum(loss_1 - self.lamda*loss_2)
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError("Invalid reduction option.")
        return loss
    
if __name__ == "__main__":
    input1 = torch.rand(1, 1, 5, 5)
    input2 = torch.rand(1, 1, 5, 5)
    criterion = SSIMLoss(5)
    loss = criterion(input1, input2)
    print(loss)

    input1 = torch.ones(1, 1, 5, 5)* 3
    input2 = torch.ones(1, 1, 5, 5)* 1
    criterion_si = ScaleInvariantLoss('cpu')
    loss_si = criterion_si(input1, input2)
    print(loss_si)