from tracksdata.nodes._mask import Mask


def _compute_iou(mask1: Mask, mask2: Mask) -> list[tuple[int, int, float]]:
    """Compute label IOUs between two Mask objects.

    Args:
        mask1 (Mask): First mask object
        mask2 (Mask): Second mask object

    Returns:
        iou (int): IOU value between the two masks
    """
    iou = mask1.iou(mask2)
    return iou
