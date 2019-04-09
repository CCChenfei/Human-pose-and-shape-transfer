

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 14. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    """
    import numpy as np
    import cv2

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple',
    ]

    if joints.shape[1] == 14:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 13, -1
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    # elif joints.shape[1] == 19:
    #     parents = np.array([
    #         1,
    #         2,
    #         8,
    #         9,
    #         3,
    #         4,
    #         7,
    #         8,
    #         -1,
    #         -1,
    #         9,
    #         10,
    #         13,
    #         -1,
    #     ])
    #     ecolors = {
    #         0: 'light_pink',
    #         1: 'light_pink',
    #         2: 'light_pink',
    #         3: 'pink',
    #         4: 'pink',
    #         5: 'pink',
    #         6: 'light_blue',
    #         7: 'light_blue',
    #         10: 'light_blue',
    #         11: 'blue',
    #         12: 'purple'
    #     }
    # else:
    #     print('Unknown skeleton!!')
    #     import ipdb
    #     ipdb.set_trace()

        for child in xrange(len(parents)):
            point = joints[:, child]
            # If invisible skip
            if vis is not None and vis[child] == 0:
                continue
            if draw_edges:
                cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                           -1)
                cv2.circle(image, (point[0], point[1]), radius - 1,
                           colors[jcolors[child]], -1)
            else:
                # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
                cv2.circle(image, (point[0], point[1]), radius - 1,
                           colors[jcolors[child]], 1)
                # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
            pa_id = parents[child]
            if draw_edges and pa_id >= 0:
                if vis is not None and vis[pa_id] == 0:
                    continue
                point_pa = joints[:, pa_id]
                cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                           colors[jcolors[pa_id]], -1)
                if child not in ecolors.keys():
                    print('bad')
                    import ipdb
                    ipdb.set_trace()
                cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                         colors[ecolors[child]], radius - 2)

        # # Convert back in original dtype
        # if input_is_float:
        #     if max_val <= 1.:
        #         image = image.astype(np.float32) / 255.
        #     else:
        #         image = image.astype(np.float32)

        return image


def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    import numpy as np
    import cv2
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = np.array([0, 0, 0])
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), 0, 0.45, black)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


def plot_j3d(j3d,j3d_gt=None):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import numpy as np
    import matplotlib.pyplot as plt
    parents = np.array([
        1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 13
    ])
    # line = [[j3d[i], j3d[parents[i]]] for i in range(13)]
    # j3d = j3d[0, 0]
    j3d=j3d-(j3d[2,:]+j3d[3,:])/2.
    j3d_gt = j3d_gt - (j3d_gt[2,:] + j3d_gt[3, :]) / 2.

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax1=fig.add_subplot(122, projection='3d')
    ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2])
    for i in range(13):
        ax.plot([j3d[i, 0], j3d[parents[i], 0]], [j3d[i, 1], j3d[parents[i], 1]], [j3d[i, 2], j3d[parents[i], 2]]
                ,color='black')
    if j3d_gt is not None:
        ax.scatter(j3d_gt[:, 0], j3d_gt[:, 1], j3d_gt[:, 2])
        for i in range(13):
            ax.plot([j3d_gt[i, 0], j3d_gt[parents[i], 0]],
                    [j3d_gt[i, 1], j3d_gt[parents[i], 1]],
                    [j3d_gt[i, 2], j3d_gt[parents[i], 2]]
                    ,color='palegreen'
                    )

            # ax.add_collection3d(Line3DCollection(line))
    plt.show()
    import ipdb
    ipdb.set_trace()
