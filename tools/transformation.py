import numpy as np

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc
def rotate_pc_along_x(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, sinval], [-sinval, cosval]])
    pc[:, [1, 2]] = np.dot(pc[:, [1, 2]], np.transpose(rotmat))
    return pc
def get_center_view_point_set(points, frustum_angle_for_y, frustum_angle_for_x):
    ''' Frustum rotation of point clouds.
    NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    '''
    frustum_angle_for_y = np.pi / 2.0 + frustum_angle_for_y
    frustum_angle_for_x = np.pi / 2.0 + frustum_angle_for_x
    # Use np.copy to avoid corrupting original data
    point_set = np.copy(points)
    return rotate_pc_along_x(rotate_pc_along_y(point_set, frustum_angle_for_y),\
        frustum_angle_for_x)