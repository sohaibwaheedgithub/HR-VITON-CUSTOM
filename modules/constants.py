hparams = {
    'cg_batch_size': 1,
    'ig_batch_size': 1,
    'cg_input_resolution': [256, 192],
    'cg_dataset_size': 33,
    'ig_dataset_size': 33,
    'cg_epochs': 100000,
    'ig_epochs': 100000,
    'lambda_ce': 10,
    'lambda_l1': 5,
    'lambda_tv': 2,
    'lambda_lsgan': 10,
    'ig_lambda_vgg_loss': 10,
    'ig_lambda_fm_loss': 10,
    'parse_agnostic_classes': 9,
    'image_parse_classes': 12,
    'data_dir': r'temp_train_2',
    'DRS_normalization_constant': 'To be Calculated'
}


lip_label_colours = [(0, 0, 0),  # 0=Background
                     (0, 85, 0),   # 1=Glove
                     (255, 85, 0),  # 2=UpperClothes
                     (85, 85, 0),  # 3=Socks
                     (0, 85, 85),  # 4=Pants
                     (0, 128, 0),  # 5=Skirt
                     (51, 170, 221),  # 6=LeftArm
                     (0, 255, 255),  # 7=RightArm
                     (85, 255, 170),  # 8=LeftLeg
                     (170, 255, 85),  # 9=RightLeg
                     (255, 255, 0),  # 10=LeftShoe
                     (255, 170, 0)  # 11=RightShoe
                     ]
