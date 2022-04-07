import os
from argparse import ArgumentParser
from glob import glob
import numpy as np
import cv2
import json

from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

import time
from scipy.optimize import least_squares

import vispy.scene
from vispy import app
from vispy.util.transforms import translate, rotate


def parse():
    """
    Parse command line
    :returns: options
    """
    parser = ArgumentParser()
    parser.add_argument('-W', '--width', default=5, help='number of rows - 1', type=int)
    parser.add_argument('-H', '--height', default=4, help='number of columns - 1', type=int)
    parser.add_argument('--grid_length', default=0.02, help='length of a single grid', type=float)
    parser.add_argument("--debug", help="debug mode", action='store_true')
    parser.add_argument('data', help='data directory', type=str)
    return parser.parse_args()

class Calibration(object):

    def __init__(self, w, h, grid_width, debug=False):
        assert w > 0
        assert h > 0
        assert grid_width > 0

        self.grid_rows = w
        self.grid_cols = h
        self.grid_width = grid_width

        self.n_cameras = 0
        self.data_dir = ""
        self.debug = debug

        self.camera_ids = []

        self.debug_colors = []

        self.init_debug_color()


    def init_debug_color(self):
        """
        Create color legend for debug purpose
        """

        self.debug_colors = []
        n_points = self.grid_cols * self.grid_rows
        for y in range(self.grid_cols):
            for x in range(self.grid_rows):
                self.debug_colors.append((255.0 / self.grid_rows * (x+1), 255 / self.grid_cols * (y+1), 0.0))


    def find_world_space_points(self):
        """
        Find the world-coordinate positions of the grid corners
        :returns: The computed points 
        """
        points = []

        for y in range(self.grid_cols):
            for x in range(self.grid_rows):
                points.append([x * self.grid_width, -y * self.grid_width, 0])
        points = np.array(points, dtype="double")


        return points


    def find_points(self, img, cam_id):
        """
        Find the image-space coordinate positions of the grid corners

        :param img: opencv-type input image

        :returns: detected points
        """

        # Preprocess an input image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find grid points
        ret, interest_points = cv2.findChessboardCorners(gray, (self.grid_rows, self.grid_cols), None)
        assert len(interest_points) == (self.grid_rows * self.grid_cols)

        # Determine the found grid points more accurately
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, interest_points, (11, 11), (-1, -1), criteria)

        # Export annotated image
        if self.debug:
            img_annotated = img.copy()
            for idx, p in enumerate(interest_points):
                px = int(p[0][0])
                py = int(p[0][1])
                img_annotated = cv2.circle(img_annotated, (px, py), 8, self.debug_colors[idx], 2)
                img_annotated = cv2.putText(img_annotated, str(idx), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(self.data_dir, 'debug', cam_id + "_points.png"), img_annotated)

        # Store detected points
        points = []
        for p in interest_points:
            assert len(p) == 1
            points.append([p[0][0], p[0][1]])

        points = np.array(points, dtype="double")

        return points


    def solve_extrinsics(self, cam_id, fx, fy, ppx, ppy):
        """
        Find the image-space coordinate positions of the grid corners

        :param data_dir: directory containing input images
        :param cam_id: image file name
        :param fx: focal length of the image in x-axis
        :param fy: focal length of the image in y-axis
        :param ppx: the pixel coordinates of the principal point in x-axis
        :param ppy: the pixel coordinates of the principal point in y-axis
        """

        img_file = os.path.join(self.data_dir, cam_id + ".png")
        img = cv2.imread(img_file)
        assert img is not None

        points_3D = self.find_world_space_points()
        points_2D = self.find_points(img, cam_id)

        # Assume no camera distortion
        dist_coeffs = np.zeros((4, 1))

        # Set camera instrinsics matrix
        camera_matrix = np.array([
                            [fx, 0, ppx],
                            [0,  fy, ppy],
                            [0,  0,  1]
                        ])

        success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags=0)
        assert success

        # Export projected results 
        if self.debug:
            p_camera, _ = cv2.projectPoints(points_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            p_camera = p_camera[:, 0, :]
            img_annotated = img.copy()
            for pc in p_camera:
                img_annotated = cv2.circle(img, (int(pc[0]), int(pc[1])), 8, (0, 255, 0), 2)

            cv2.imwrite(os.path.join(self.data_dir, 'debug', cam_id + "_projected.png"), img_annotated)

        return rotation_vector, translation_vector, points_2D



    # ----------------------------------------------------------------------------------------------------
    # Bundle Adjustment 
    # ----------------------------------------------------------------------------------------------------
    def project(self, points_3d, camera_params, camera_intrinsics, point_indices, camera_indices):
        """
        Project 3D points onto 2D camera space

        :param points_3d: An array of 3D points
        :param camera_params: 2D array of camera parameters = (camera id, [rotation|translation])
        :param camera_intrinsics: 2D array of camera intrinsics = (camera id, [fx, fy, ppx, ppy])
        :param point_indices: Indices of 3D points
        :param camera_indices: Indices of cameras for each entry in points_3d

        :returns: An array of the projected 2D coordinates
        """
        p_projected = []

        for i_cam in range(4):

            fx = camera_intrinsics[i_cam][0]
            fy = camera_intrinsics[i_cam][1]
            ppx = camera_intrinsics[i_cam][2]
            ppy = camera_intrinsics[i_cam][3]
            camera_matrix = np.array([
                                    [fx, 0, ppx],
                                    [0,  fy, ppy],
                                    [0,  0,  1]
                                ])

            rotation_vector = camera_params[i_cam, :3]
            translation_vector = camera_params[i_cam, 3:6]

            dist_coeffs = np.zeros((4, 1))

            p_camera, _ = cv2.projectPoints(points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            p_projected.append(p_camera)

        p_projected = np.array(p_projected)
        p_projected = p_projected.reshape((80, 2))

        return p_projected

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_intrinsics):
        """Compute residuals.
        
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d, camera_params, camera_intrinsics, point_indices, camera_indices)
        return (points_proj - points_2d).ravel()


    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)


        i = np.arange(camera_indices.size)


        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            for s_i in (n_cameras * 6 + point_indices * 3 + s):
                A[2 * i, s_i] = 1

            for s_i in (n_cameras * 6 + point_indices * 3 + s):
                A[2 * i + 1, s_i] = 1

        return A

    def render(self, points_3d, camera_params):
        """
        Render grids and camera positions

        :param points_3d: An array of 3D points
        :param camera_params: 2D array of camera parameters = (camera id, [rotation|translation])
        """

        # Initialize canvas
        c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = c.central_widget.add_view()
        axis = vispy.scene.visuals.XYZAxis(parent=view.scene)
        view.camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=71.5, elevation=84.5, distance=1, up='+z')

        # Plot grid points with color
        for idx, gp in enumerate(points_3d):
            checkerboard_points = vispy.scene.visuals.Markers()
            checkerboard_points.antialias = 0
            color = tuple(dc/255.0 for dc in self.debug_colors[idx])
            color = (color[2], color[1], color[0])
            checkerboard_points.set_data(np.array([gp]), edge_color='black', face_color=color)
            view.add(checkerboard_points)


        # Render camera
        for idx, cp in enumerate(camera_params):
            rotation_vec = cp[:3]
            translation_vec = cp[3:]

            rot_max, _ = cv2.Rodrigues(rotation_vec)
            rot_max_inv = rot_max.transpose()

            camera_pos_world = np.matmul(-rot_max_inv, translation_vec)

            # print(camera_pos_world)
            camera_pos_world = np.array([camera_pos_world])

            cam_p = vispy.scene.visuals.Markers()
            cam_p.antialias = 0
            cam_p.set_data(camera_pos_world, edge_color='black', face_color='white')
            view.add(cam_p)

            pos = tuple(camera_pos_world[0])

            texts = vispy.scene.visuals.Text(self.camera_ids[idx], pos=pos, font_size=6, color=(1,0,0), parent=view.scene)

        app.run()


    def solve(self, data_dir):
        """
        Run multi-camera calibration and write out extrinsics.json file
        containing camera extrinsics

        :param data_dir: Directory containing captured images and intrinsics.json
        """

        self.data_dir = data_dir

        if self.debug:
            os.makedirs(os.path.join(self.data_dir, "debug"), exist_ok = True)

        # Open intrinsics.json file in the directory
        try:
            with open(os.path.join(self.data_dir, "intrinsics.json"), 'r') as f:
                cams_intrinsics = json.load(f)
        except:
            sys.exit("intrinsics.json not found")


        p_2ds = []
        c_vecs = []
        c_intrinsics = []
        self.camera_ids = list(cams_intrinsics.keys())
        extrinsics = {}


        """
        Run single-view calibration to find the initial state for
        the 3D grid points, rotation and translation matrix for each camera
        """
        for cam_id in self.camera_ids:

            intrinsics = cams_intrinsics[cam_id]
            fx = intrinsics[0]
            fy = intrinsics[1]
            ppx = intrinsics[2]
            ppy = intrinsics[3]

            c_intrinsics.append(np.array([fx, fy, ppx, ppy]))

            rotation_vector, translation_vector, points_2D = self.solve_extrinsics(cam_id, fx, fy, ppx, ppy)

            p_2ds.append(points_2D)
            c_vecs.append(rotation_vector)
            c_vecs.append(translation_vector)


        """
        Run bundle adjustment to refine 3D points and camera matrices (rotation and translation)
        """
        n_cameras = len(self.camera_ids)
        n_points = self.grid_rows * self.grid_cols

        points_3d = self.find_world_space_points()
        points_3d = points_3d.ravel()

        camera_indices = np.floor(np.arange(0, n_cameras * n_points) / n_points)
        point_indices = np.arange(0, n_points)

        points_2d = np.array(p_2ds)
        points_2d = points_2d.reshape((-1,2))

        camera_params = np.array(c_vecs).ravel()

        c_intrinsics = np.array(c_intrinsics)

        n = 6 * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        if self.debug:
            print("n_cameras: {}".format(n_cameras))
            print("n_points: {}".format(n_points))
            print("Total number of parameters: {}".format(n))
            print("Total number of residuals: {}".format(m))
    

        x0 = np.hstack((camera_params, points_3d))
        f0 = self.fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, c_intrinsics)

        if self.debug:
            plt.plot(f0)

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)


        verbose = 2 if self.debug else 0
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, c_intrinsics))


        if self.debug:
            plt.plot(res.fun)
            plt.savefig(os.path.join(self.data_dir, 'debug/errors_opt.png'))


        camera_params_opt = res.x[:4*6]
        points_3d_opt = res.x[4*6:]

        camera_params_opt = camera_params_opt.reshape(-1, 6)
        points_3d_opt = points_3d_opt.reshape(-1, 3)


        """
        Show debug info of the optimization result
        """
        if self.debug:
            print("Grid points on chessboard before optimization")
            print(points_3d)
            print("Grid points on chessboard after optimization")
            print(points_3d_opt)
            print("Differences between before-opt and after-opt")
            print(self.find_world_space_points() - points_3d_opt)


        """
        Render 3D visualization of the chessboard and camera positions 
        """
        if self.debug:
            self.render(points_3d_opt, camera_params_opt)



        """
        Export exntrinsics.json
        """
        extrinsics = {}

        # Render camera
        for idx, cp in enumerate(camera_params_opt):
            rotation_vec = cp[:3]
            translation_vec = cp[3:]

            rot_max, _ = cv2.Rodrigues(rotation_vec)
            rot_max_inv = rot_max.transpose()

            camera_pos_world = np.matmul(-rot_max_inv, translation_vec)

            # print(camera_pos_world)
            camera_pos_world = np.array([camera_pos_world])


            extrinsics[cam_id] = {
                "rotation": rotation_vector.flatten().tolist(),
                "translation": translation_vector.flatten().tolist(),
                "camera_world": camera_pos_world.flatten().tolist()
            }

        with open('data/extrinsics.json', 'w') as outfile:
            json.dump(extrinsics, outfile)



def main():
    opts = parse()

    calibration = Calibration(opts.width, opts.height, opts.grid_length, debug=opts.debug)
    calibration.solve(opts.data)


if __name__ == "__main__":
    main()