import numpy as np
from scipy import optimize as opt

def normalisation_matrix(flattened_corners):
    avg_x = flattened_corners[:, 0].mean()
    avg_y = flattened_corners[:, 1].mean()

    s_x = np.sqrt(2 / flattened_corners[0].std())
    s_y = np.sqrt(2 / flattened_corners[1].std())

    return np.matrix([
        [s_x,   0,   -s_x * avg_x],
        [0,   s_y,   -s_y * avg_y],
        [0,     0,              1]
    ])


def initial_guess(first, second):
    first_normalisation_matrix = normalisation_matrix(first)
    second_normalisation_matrix = normalisation_matrix(second)

    M = []

    for j in range(0, first.size / 2):
        homogeneous_first = np.array([
            first[j][0],
            first[j][1],
            1
        ])

        homogeneous_second = np.array([
            second[j][0],
            second[j][1],
            1
        ])

        pr_1 = np.dot(first_normalisation_matrix, homogeneous_first)
        pr_2 = np.dot(second_normalisation_matrix, homogeneous_second)

        M.append(np.array([
            pr_1.item(0), pr_1.item(1), 1,
            0, 0, 0,
            -pr_1.item(0)*pr_2.item(0), -pr_1.item(1)*pr_2.item(0), -pr_2.item(0)
        ]))

        M.append(np.array([
            0, 0, 0, pr_1.item(0), pr_1.item(1),
            1, -pr_1.item(0)*pr_2.item(1), -pr_1.item(1)*pr_2.item(1), -pr_2.item(1)
        ]))

    U, S, Vh = np.linalg.svd(np.array(M).reshape((512, 9)))
    L = Vh[-1]
    H = L.reshape(3, 3)

    denormalised = np.dot(
        np.dot(
            np.linalg.inv(first_normalisation_matrix),
            H
        ),
        second_normalisation_matrix
    )
    return denormalised / denormalised[-1, -1]


def find_camera_matrix(matrix, data):
    [world_cords, image_points] = data

    Y = []

    for i in range(0, world_cords.size / 2):
        x = world_cords[i][0]
        y = world_cords[i][1]

        w = matrix[6] * x + matrix[7] * y + matrix[8]

        M = np.array([
            [matrix[0], matrix[1], matrix[2]],
            [matrix[3], matrix[4], matrix[5]]
        ])

        homog = np.transpose(np.array([x, y, 1]))
        [u, v] = (1/w) * np.dot(M, homog)

        Y.append(u)
        Y.append(v)

    return np.array(Y)


def jacobian(matrix, data):
    [world_cords, image_points] = data

    J = []
    for i in range(0, world_cords.size / 2):
        x = world_cords[i][0]
        y = world_cords[i][1]

        s_x = matrix[0] * x + matrix[1] * y + matrix[2]
        s_y = matrix[3] * x + matrix[4] * y + matrix[5]
        w = matrix[6] * x + matrix[7] * y + matrix[8]

        J.append(
            np.array([
                x / w, y / w, 1/w,
                0, 0, 0,
                (-s_x * x) / (w*w), (-s_x * y) / (w*w), -s_x / (w*w)
            ])
        )

        J.append(
            np.array([
                0, 0, 0,
                x / w, y / w, 1 / w,
                (-s_y * x) / (w*w), (-s_y * y) / (w*w), -s_y / (w*w)
            ])
        )

    return np.array(J)


def refine_camera_matrix(matrix, world_cords, image_points):
    return opt.root(
        find_camera_matrix,
        matrix,
        jac=jacobian,
        args=[world_cords, image_points],
        method='lm'
    ).x


def compute_camera_matrix(data):
    image_points = data['image_points']
    refined_homographies = []

    for i in range(0, len(data['world_cords'])):
        world_cords = data['world_cords'][i]
        estimated = initial_guess(image_points, world_cords)
        
        print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
        print estimated.shape
        refined = refine_camera_matrix(estimated, world_cords, image_points)
        refined = refined / refined[-1]

        print refined.shape
        refined_homographies.append(refined)

    return np.array(refined_homographies)

# supply input data 
