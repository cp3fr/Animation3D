# Animation3D

3D Animation of flight trajectories


## Input data

The class Animation3D takes as input a pandas dataframe with columns:

__'ts'__ : Timestamp in sec

__'PositionX', 'PositionY', 'PositionZ'__ :

Quadrotor position in meters in world frame (x=forward, y=left, z=up)

__'rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat'__ :

Quadrotor rotation quaternion in world frame (uses scipy .from_quat())

## Example

Run the script `src/demo.py`, which give the following output:

![](/media/anim.gif)
