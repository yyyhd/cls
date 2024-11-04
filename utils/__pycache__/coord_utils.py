import numpy as np


def centerd_voxel_to_world(centerd, origin_xyz, spacing_xyz, direction_xyz):
    ''' Convert centerd voxel coords to world coords.
    centerd: ndarray, with shape of [N, M], columns of [x, y, z, dx, dy, dz, ...]
    '''
    origin_xyz = np.array(origin_xyz)
    spacing_xyz = np.array(spacing_xyz)
    direction_xyz = np.array(direction_xyz)

    centers_xyz = centerd[:, :3]
    centers_xyz = centers_xyz * spacing_xyz * direction_xyz + origin_xyz

    diameters_xyz = centerd[:, 3:6]
    diameters_xyz = diameters_xyz * spacing_xyz

    arr_new = np.concatenate([centers_xyz, diameters_xyz, centerd[:, 6:]], axis=1)
    return arr_new


def voxel_coord_2_world(voxelCoord, origin, spacing, directions=None):
    """
    图像坐标转世界坐标
    注意:ImageOrientationPatient的tag只考虑了全为1或全为-1的情况
    """
    if directions is None:
        directions = np.array([1] * len(voxelCoord))
    voxelCoord, origin, spacing, directions = np.array(voxelCoord), np.array(origin), np.array(spacing), np.array(
        directions)
    stretchedVoxelCoord = np.array(voxelCoord) * np.array(spacing).astype(float)
    worldCoord = np.array(origin) + stretchedVoxelCoord * np.array(directions)
    return worldCoord


def get_world_coord(bboxes, infos):
    origin_zyx = infos['origin']
    spacing_zyx = infos['new_spacing']
    direction_zyx = infos['direction']
    bboxes[:, :3] = voxel_coord_2_world(bboxes[:, :3], origin_zyx, spacing_zyx, direction_zyx)
    bboxes[:, 3:6] = voxel_coord_2_world(bboxes[:, 3:6], origin_zyx, spacing_zyx, direction_zyx)
    return bboxes


def centerd2bbox(center, diams, image_shape=None):
    center = np.array(center, dtype=float)
    diams = np.array(diams, dtype=float)
    start = center - diams / 2
    end = center + diams / 2
    if image_shape != None:
        start[start < 0] = 0
        end[end > np.array(image_shape)] = np.array(image_shape)[end > np.array(image_shape)]
    return start, end


def is_rect_overlap(rect1_start, rect1_end, rect2_start, rect2_end):
    if rect1_start[0] > rect2_end[0] or rect1_start[1] > rect2_end[1] or rect2_start[0] > rect1_end[0] or rect2_start[1] > rect1_end[1] or rect1_start[2] > rect2_end[2] or rect2_start[2] > rect1_end[2]:
        isOverlap = -1
    else:
        rect1_start = np.array(rect1_start)
        rect1_end = np.array(rect1_end)
        rect2_start = np.array(rect2_start)
        rect2_end = np.array(rect2_end)

        area1 = rect1_end - rect1_start
        area1 = area1[0] * area1[1] * (area1[2] + 1)

        area2 = rect2_end - rect2_start
        area2 = area2[0] * area2[1] * (area2[2] + 1)

        intersection = np.min([rect1_end, rect2_end], axis=0) - np.max([rect1_start, rect2_start], axis=0)
        intersection = intersection[0] * intersection[1] * (intersection[2] + 1)

        isOverlap = abs(intersection / (area1 + area2 - intersection))
    return isOverlap

