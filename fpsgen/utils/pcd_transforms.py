import numpy as np

def rotate_point_cloud_90(batch_data):
    """Rotate point cloud by 0, 90, 180, or 270 degrees around Z-axis."""
    rotated_data = np.zeros_like(batch_data)
    for i in range(batch_data.shape[0]):
        angle = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
        cosval, sinval = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval,  cosval, 0],
                                    [0, 0, 1]])
        rotated_data[i] = np.dot(batch_data[i], rotation_matrix)
    return rotated_data

def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_translate_point_cloud_perp(points, sigma=0.4):
    """
     batch
    points: (B, N, 3)
    sigma:
    """
    B, N, C = points.shape
    translation = np.zeros((B, 1, 3))
    for b in range(B):
        p0 = points[b, 0, :2]
        p1 = points[b, 1, :2]
        tangent = p1 - p0
        if np.linalg.norm(tangent) < 1e-8:
            tangent = np.array([1.0, 0.0])
        tangent = tangent / np.linalg.norm(tangent)
        perp = np.array([-tangent[1], tangent[0]])
        offset = np.random.normal(loc=0.0, scale=sigma / 2)
        offset = np.clip(offset, -sigma, sigma)
        translation[b, 0, :2] = perp * offset
    return points + translation
def rotate_point_cloud_ros(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform(-np.pi/4, np.pi/4)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval,  cosval, 0],
                                    [0,       0,      1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud_xy(batch_data, sigma=0.01, clip=0.05):
    """
     XY ，Z
    batch_data: (B, N, 3)
    sigma:
    clip:
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jitter_xy = np.clip(sigma * np.random.randn(B, N, 2), -clip, clip)
    jitter = np.zeros_like(batch_data)
    jitter[:, :, :2] = jitter_xy
    jittered_data = batch_data + jitter
    return jittered_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def random_drop_n_cuboids(batch_data):
    batch_data = random_drop_point_cloud(batch_data)
    cuboids_count = 1
    while cuboids_count < 5 and np.random.uniform(0., 1.) > 0.3:
        batch_data = random_drop_point_cloud(batch_data)
        cuboids_count += 1

    return batch_data

def check_aspect2D(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2])/np.max(crop_range[:2])
    return (xy_aspect >= aspect_min)

def random_cuboid_point_cloud(batch_data):
    batch_data = np.expand_dims(batch_data, axis=0)

    B, N, C = batch_data.shape
    new_batch_data = []
    for batch_index in range(B):
        range_xyz = np.max(batch_data[batch_index,:,0:2], axis=0) - np.min(batch_data[batch_index,:,0:2], axis=0)

        crop_range = 0.5 + (np.random.rand(2) * 0.5)

        loop_count = 0
        while not check_aspect2D(crop_range, 0.75):
            loop_count += 1
            crop_range = 0.5 + (np.random.rand(2) * 0.5)
            if loop_count > 100:
                break

        loop_count = 0

        while True:
            loop_count += 1
            new_range = range_xyz * crop_range / 2.0
            sample_center = batch_data[batch_index,np.random.choice(len(batch_data[batch_index])), 0:3]
            max_xyz = sample_center[:2] + new_range
            min_xyz = sample_center[:2] - new_range

            upper_idx = np.sum((batch_data[batch_index,:,:2] < max_xyz).astype(np.int32), 1) == 2
            lower_idx = np.sum((batch_data[batch_index,:,:2] > min_xyz).astype(np.int32), 1) == 2

            new_pointidx = ((upper_idx) & (lower_idx))

            if (loop_count > 100) or (np.sum(new_pointidx) > 20000):
                break


        new_batch_data.append(batch_data[batch_index,new_pointidx,:])

    new_batch_data = np.array(new_batch_data)

    return np.squeeze(new_batch_data, axis=0)

def random_drop_point_cloud(batch_data):
    B, N, C = batch_data.shape
    new_batch_data = []
    for batch_index in range(B):
        range_xyz = np.max(batch_data[batch_index,:,0:3], axis=0) - np.min(batch_data[batch_index,:,0:3], axis=0)

        crop_range = np.random.uniform(0.1, 0.15)
        new_range = range_xyz * crop_range / 2.0
        sample_center = batch_data[batch_index,np.random.choice(len(batch_data[batch_index])), 0:3]
        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = np.sum((batch_data[batch_index,:,0:3] < max_xyz).astype(np.int32), 1) == 3
        lower_idx = np.sum((batch_data[batch_index,:,0:3] > min_xyz).astype(np.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))
        new_batch_data.append(batch_data[batch_index,new_pointidx,:])

    return np.array(new_batch_data)


def random_flip_point_cloud(batch_data, scale_low=0.95, scale_high=1.05):
    B, N, C = batch_data.shape
    for batch_index in range(B):
        if np.random.random() > 0.5:
            batch_data[batch_index,:,1] = -1 * batch_data[batch_index,:,1]
    return batch_data

def random_scale_point_cloud(batch_data, scale_low=0.95, scale_high=1.05):
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:]
    return batch_pc
