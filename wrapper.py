# demo.py ---
#
# Filename: demo.py
# Description: Demo of the 3DSmoothNet pipeline. 
# Comment: Some functions adapated from the open3d library http://www.open3d.org/
#
# Author: Gojcic Zan, Zhou Caifa
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Paper: https://arxiv.org/abs/1811.06879
# Created: 03.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:


import tensorflow as tf
import copy
import numpy as np
import os
import subprocess
from open3d import *


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def execute_global_registration(
        source_down, target_down, reference_desc, target_desc, distance_threshold):

#    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#            source_down, target_down, reference_desc, target_desc,
#            True, distance_threshold)

    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, reference_desc, target_desc,
            True, distance_threshold,
            open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            checkers=[],
            criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(4000000))

#    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#            source_down, target_down, reference_desc, target_desc,
#            False, distance_threshold,
#            open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
#            [open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#            open3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = open3d.pipelines.registration.registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            open3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == '__main__':
    # Run the input parametrization
    point_cloud_files = ["./data/demo/cloud_bin_0.ply", "./data/demo/cloud_bin_1.ply"]
    keypoints_files = ["./data/demo/cloud_bin_0_keypoints.txt", "./data/demo/cloud_bin_1_keypoints.txt"]



    for i in range(0,len(point_cloud_files)):
        args = "./3DSmoothNet -f " + point_cloud_files[i] + " -k " + keypoints_files[i] +  " -o ./data/demo/sdv/"
#        args = "./3DSmoothNet -f " + point_cloud_files[i] + " -k 0" +  " -o ./data/demo/sdv/"
        subprocess.call(args, shell=True)

    print('Input parametrization complete. Start inference')


    # Run the inference as shell
    args = "python main_cnn.py --run_mode=test --evaluate_input_folder=./data/demo/sdv/  --evaluate_output_folder=./data/demo"
    subprocess.call(args, shell=True)

    print('Inference completed perform nearest neighbor search and registration')


    # Load the descriptors and estimate the transformation parameters using RANSAC
    reference_desc = np.load('./data/demo/32_dim/cloud_bin_0.ply_0.150000_16_1.750000_3DSmoothNet.npz')
    reference_desc = reference_desc['data']


    test_desc = np.load('./data/demo/32_dim/cloud_bin_1.ply_0.150000_16_1.750000_3DSmoothNet.npz')
    test_desc = test_desc['data']

    # Save as open3d feature
    ref = open3d.pipelines.registration.Feature()
    ref.data = reference_desc.T

    test = open3d.pipelines.registration.Feature()
    test.data = test_desc.T

    # Load point cloud and extract the keypoints
    reference_pc = open3d.io.read_point_cloud(point_cloud_files[0])
    test_pc = open3d.io.read_point_cloud(point_cloud_files[1])

    indices_ref = np.genfromtxt(keypoints_files[0])
    indices_test = np.genfromtxt(keypoints_files[1])

    reference_pc_keypoints = np.asarray(reference_pc.points)[indices_ref.astype(int),:]
    test_pc_keypoints = np.asarray(test_pc.points)[indices_test.astype(int),:]


    # Save ad open3d point clouds
    ref_key = open3d.geometry.PointCloud()
    ref_key.points = open3d.utility.Vector3dVector(reference_pc_keypoints)

    test_key = open3d.geometry.PointCloud()
    test_key.points = open3d.utility.Vector3dVector(test_pc_keypoints)

    result_ransac = execute_global_registration(ref_key, test_key,
                ref, test, 0.05)


    # First plot the original state of the point clouds
    # draw_registration_result(reference_pc, test_pc, np.identity(4))


    # Plot point clouds after registration
    print(result_ransac)
    #draw_registration_result(reference_pc, test_pc,
    #            result_ransac.transformation)

    if len(result_ransac.correspondence_set) == 0:
        tsfm = np.zeros((4, 4))
    else:
        tsfm = result_ransac.transformation
    save_result_path = "./data/demo"
    with open(os.path.join(save_result_path, "result.txt"), 'w') as f:
        for idx in range(tsfm.shape[0]):
            for i in range(0, 4):
                f.write(str(tsfm[idx, i]) + "\t")
            f.write('\n')


