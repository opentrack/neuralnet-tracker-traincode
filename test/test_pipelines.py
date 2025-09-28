import torch
import os
import pytest

from trackertraincode.pipelines import make_pose_estimation_loaders, Id


@pytest.mark.skipif("DATADIR" not in os.environ, reason="DATADIR is not set")
def test_make_pose_estimation_loaders():
    train_loader, test_loader, _ = make_pose_estimation_loaders(
        inputsize=129,
        batchsize=9,
        datasets=[Id._300WLP],
        dataset_weights=None,
        use_weights_as_sampling_frequency=True,
        enable_image_aug=True,
        rotation_aug_angle=0.0,
        roi_override="original",
        device="cpu",
    )

    for _, batches in zip(range(10), train_loader):
        for batch in batches:
            assert "image" in batch
            assert "pose" in batch
            assert "coord" in batch
            assert "pt3d_68" in batch
            assert "roi" in batch
            assert batch["image"].device == torch.device("cpu")

    for _, batch in zip(range(10), test_loader):
        assert "image" in batch
        assert "pose" in batch
        assert "coord" in batch
        assert "pt3d_68" in batch
        assert "roi" in batch
        assert batch["image"].device == torch.device("cpu")
