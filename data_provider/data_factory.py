from torch.utils.data import DataLoader

from data_provider.data_loader import DatasetWideForecast


def data_provider(args, flag):
    shuffle_flag = flag == "train"
    drop_last = flag == "train"
    batch_size = args.batch_size

    data_set = DatasetWideForecast(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        features=args.features,
        target=args.target,
        time_col=args.time_col,
        freq=args.freq,
        wind_dir_encoding=args.wind_dir_encoding,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        data_limit=args.data_limit,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=args.use_gpu,
    )
    return data_set, data_loader
