from torch.utils.data import DataLoader

from datasets.build import build_test_dataset, build_train_dataset


def get_train_dataloader(cfg, train_TN):
    ds = build_train_dataset(cfg, train_TN)
    return DataLoader(
        ds,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        drop_last=cfg.TRAIN.DROP_LAST,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )


def get_test_dataloader(cfg, test_TN):
    ds = build_test_dataset(cfg, test_TN)
    return DataLoader(
        ds,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )
