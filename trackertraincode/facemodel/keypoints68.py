'''
Because keypoints are localized in a specific place in the face, we must
reorder keypoints after a horizontal flip. I.e. left and right sides must
be exchanged.
'''
flip_map = [
    16,
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    0,
    26,
    25,
    24,
    23,
    22,
    21,
    20,
    19,
    18,
    17,
    27,
    28,
    29,
    30,
    35, # 31
    34,
    33,
    32,
    31,
    45, # 36
    44,
    43,
    42, # 39
    47,
    46, # 41
    39,
    38,
    37,
    36,
    41, 
    40, # 47
    54,
    53,
    52,
    51,
    50,
    49,
    48,
    59,
    58,
    57,
    56,
    55,
    64, # 60
    63,
    62,
    61,
    60,
    67, # 65
    66,
    65,
]

# Warning: both sides contain middle points
chin_left = [*range(0,9)]
chin_right = [*range(8,17)]

upperlip_left = [48, 49, 50, 51]
upperlip_right = [51,52,53,54]
lowerlip_left = [48, 59, 58, 57]
lowerlip_right = [57, 56, 55, 54]
uppermouth_left = [60,61,62]
uppermouth_right = [62,63,64]
lowermouth_left = [60,67,66]
lowermouth_right = [66,65,64]

nose_left=[31,32,33]
nose_right=[33,34,35]
nose_back=[27,28,29,30,33]

eyecorners_left=[36,39]
eyecorners_right=[42,45]
brows_left=[*range(17,22)]
brows_right=[*range(22,27)]

eye_left_top = [ 36, 37, 38, 39 ]
eye_left_bottom = [ 36, 41, 40, 39 ]

eye_right_top = [ 42, 43, 44, 45 ]
eye_right_bottom = [ 42,47,46,45 ]

eye_not_corners = [ 37, 38, 41, 40, 43, 44, 47, 46 ]

nose_tip = 33
mouth_corner_left = 60
mouth_corner_right = 64