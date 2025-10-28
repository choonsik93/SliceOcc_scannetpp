import open3d as o3d
import numpy as np
import imageio

def visualize_and_save_mesh(ply_file_path, output_image_path):
   
    mesh = o3d.io.read_triangle_mesh(ply_file_path)

    
    mesh_center = mesh.get_center()
    
    R = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Mesh Visualization', width=800, height=600)
    vis.add_geometry(mesh)

    
    ctr = vis.get_view_control()
    lookat_point = mesh_center
    front_vector = np.array([np.cos(np.deg2rad(40)), 0, np.sin(np.deg2rad(40))])
    front_vector = front_vector / np.linalg.norm(front_vector)
    up_vector = np.array([0, 1, 0])

    ctr.set_front(front_vector.tolist())
    ctr.set_lookat(lookat_point.tolist())
    ctr.set_up(up_vector.tolist())

    
    rotation_angle = np.pi / 180  

    def rotate_around_axis(vis, axis):
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        R = o3d.geometry.get_rotation_matrix_from_xyz((rotation_angle * axis[0], rotation_angle * axis[1], rotation_angle * axis[2]))
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R
        cam_params.extrinsic = np.dot(R_4x4, cam_params.extrinsic)
        ctr.convert_from_pinhole_camera_parameters(cam_params)
        vis.update_renderer()
        return False  

    def rotate_around_x(vis):
        return rotate_around_axis(vis, [1, 0, 0])

    def rotate_around_y(vis):
        return rotate_around_axis(vis, [0, 1, 0])

    def rotate_around_z(vis):
        return rotate_around_axis(vis, [0, 0, 1])

    def capture_image(vis):
        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.array(image) * 255
        imageio.imwrite(output_image_path, image.astype(np.uint8))
        print(f"Screenshot saved to {output_image_path}")
        return False  


    vis.register_key_callback(ord('X'), rotate_around_x)
    vis.register_key_callback(ord('Y'), rotate_around_y)
    vis.register_key_callback(ord('Z'), rotate_around_z)
    vis.register_key_callback(ord('S'), capture_image)


    vis.run()
    vis.destroy_window()



ply_file_path = '/mnt/data/ljn/datasets/embodiedscan/Scannet/scans/scene0291_00/scene0291_00_vh_clean.ply'
output_image_path = './scene291.png'
visualize_and_save_mesh(ply_file_path, output_image_path)