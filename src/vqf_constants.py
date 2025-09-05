
target_classes = ['chair', 'bed', 'toilet', 'couch', 'refrigerator']
target_cls_id_in_scene = [0, 1, 3, 4, 9]

target_coco_categories = {
    "chair": 0,
    "couch": 1,
    # "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    # "tv": 5,
    # "dining-table": 6,
    # "oven": 7,
    # "sink": 8,
    "refrigerator": 9,
    # "book": 10,
    # "clock": 11,
    # "vase": 12,
    # "cup": 13,
    # "bottle": 14
}

target_coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    72: 2,  # refrigerator
    59: 3,  # bed
    61: 4,  # toilet
    
    # 62: 5,  # tv
    # 60: 6,  # dining-table
    # 69: 7,  # oven
    # 71: 8,  # sink
    
    # 73: 10,  # book
    # 74: 11,  # clock
    # 75: 12,  # vase
    # 41: 13,  # cup
    # 39: 14,  # bottle
}

category_maps = {0: "chair", 1:"couch", 3:"bed", 4:"toilet", 9:"refrigerator"}
category_id_maps = {0: 56, 1:57, 3:59, 4:61, 9:72}

color_palette_vsqf = [      # rgb
    1., 1., 1.,     # 白色背景
    0.0416, 0., 0., 
    0.22690719, 0., 0.,
    0.41221439, 0., 0.,
    0.59752158, 0., 0.,
    0.79312362, 0., 0.,
    0.97843081, 0., 0.,
    1., 0.16372618, 0.,
    1., 0.3593141,  0.,
    1., 0.54460792, 0.,
    1., 0.72990173, 0.,
    1., 0.91519555, 0.,
    ]