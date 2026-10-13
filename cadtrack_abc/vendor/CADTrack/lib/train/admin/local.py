class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/wangwj/CADTrack-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/wangwj/CADTrack-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/wangwj/CADTrack-main/pretrained_networks'
        self.got10k_val_dir = '/data/wangwj/CADTrack-main/got10k/val'
        self.lasot_lmdb_dir = '/data/wangwj/CADTrack-main/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/wangwj/CADTrack-main/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/wangwj/CADTrack-main/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/wangwj/CADTrack-main/coco_lmdb'
        self.coco_dir = '/data/wangwj/CADTrack-main/coco'
        self.lasot_dir = '/data/wangwj/CADTrack-main/lasot'
        self.got10k_dir = '/data/wangwj/CADTrack-main/got10k/train'
        self.trackingnet_dir = '/data/wangwj/CADTrack-main/trackingnet'
        self.depthtrack_dir = '/data/wangwj/CADTrack-main/depthtrack/train'
        self.lasher_dir = '/data/wangwj/dataset/LasHeR/trainingset'
        self.visevent_dir = '/data/wangwj/CADTrack-main/visevent/train'
