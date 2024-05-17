import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.srv import GetEntityState, SetEntityState

from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import (Point, Pose, PoseArray, PoseStamped, Quaternion,
                               Twist, Vector3)
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from scipy.spatial.transform import Rotation as R
import numpy as np
from rclpy.task import Future
import cv2
import os
import open3d as o3d
import scipy.io as sio
import copy
import random

import time

model_workfolder = '/home/alexmanson/Documents/stanford/gazebo_ycb/models/'
#model_workfolder = '/home/alexmanson/Documents/stanford/dg_ws/src/data_generation/models/'
#region Utils Function
#%% Function to convert between Image types for depth images
def convert_types(img, orig_min, orig_max, tgt_min, tgt_max, tgt_type):

    #info = np.finfo(img.dtype) # Get the information of the incoming image type
    # normalize the data to 0 - 1
    img_out = img / (orig_max-orig_min)   # Normalize by input range
    img_out = (tgt_max - tgt_min) * img_out # Now scale by the output range
    img_out = img_out.astype(tgt_type)

    #cv2.imshow("Window", img)
    return img_out

#%% Function to fill empty spaces in point cloud project
def cv_im_fill(bin_im):
    im_floodfill = bin_im.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels smaller than the image.
    h, w = bin_im.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = bin_im | im_floodfill_inv
    return im_out

#%% Get Camera Extrinsics
def get_camera_extrinsics(phi,theta, dist):
    
    _,_,X,Y,Z,_ = calc_params(phi,theta,dist)

    cam_euler = R.from_euler('xyz',[0,phi,theta+180], degrees=True)
    cam_world_R = cam_euler.as_matrix()
    cam_world_t = np.array([X,Y,Z]).reshape(3,1)
    cam_world_T = np.hstack((cam_world_R,cam_world_t))
    cam_world_T = np.vstack((cam_world_T, [0,0,0,1]))
    
    return cam_world_T

#%% Calculate parameters
def calc_params(phi,theta,dist):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    X = dist*np.cos(phi_rad)*np.cos(theta_rad)
    Y = dist*np.cos(phi_rad)*np.sin(theta_rad)
    Z = np.abs(dist*np.sin(phi_rad)) + 0.84

    cam_euler = R.from_euler('xyz',[0,phi,theta+180], degrees=True)
    cam_quat = cam_euler.as_quat()
    
    camPos = Pose(position= Point(x=X, y=Y, z=Z), 
                orientation= Quaternion(x=cam_quat[0], y=cam_quat[1] , z=cam_quat[2], w=cam_quat[3]))
    camTwist = Twist(linear= Vector3(x=0., y=0., z=0.) , 
                    angular= Vector3(x=0., y=0., z=0.))
    
    return camPos,camTwist, X, Y,Z, cam_euler 

#%% True Object pose in world frame obtained from Gazebo Service
def get_object2cam_pose(resp, n_models, cam_world_T):
    obj_cam_T = np.zeros((  4, 4, n_models )) # Transformation Mats for 10 object classes
    
    for i in range(0 , n_models):
        obj_pos = np.array([resp[i].pose.position.x, resp[i].pose.position.y, resp[i].pose.position.z]).reshape(3,1)
        obj_or = [resp[i].pose.orientation.x, resp[i].pose.orientation.y, resp[i].pose.orientation.z, resp[i].pose.orientation.w]
        obj_or = (R.from_quat(obj_or)).as_matrix()
        obj_world_T = np.concatenate((obj_or, obj_pos), axis = 1) 

        # Transformation from object2world to  object2cam for GT label poses
        #obj_cam_T = np.dot(obj_world_T, np.linalg.inv(cam_world_T) )
        obj_world_T = np.vstack(( obj_world_T, [0,0,0,1] ))    
        obj_cam_T[:, :, i] = np.dot( np.linalg.inv(cam_world_T), obj_world_T )#[:3,:]
    
    gt_dict = { 'poses':obj_cam_T[:3,:,:] } #returns [ R  T , i] 

    return  gt_dict, obj_cam_T


#%% Transform the pointclouds to binary mask
def pcl_2_binary_mask(obj_cam_T,n_models, all_points, cam_info_msg,models):
    
    print(cam_info_msg)
    #Projection Matrix / Camera instrinsics
    cam_P = np.array(cam_info_msg.p.reshape(3,4))
    cam_P = np.vstack((cam_P , [0,0,0,1]))
    bin_mask = np.zeros((cam_info_msg.height, cam_info_msg.width), dtype= np.uint8)
    mask_list = []
    pixels_list = []
    
    #Camera optical link 
    cam2optical = R.from_euler('zyx',[1.57, 0, 1.57])
    cam2optical = cam2optical.as_matrix()
    op2cam_T = np.hstack(( np.vstack(( cam2optical , [0,0,0] )) , np.array([[0],[0],[0],[1]]) ))
    
    
    for i in np.argsort(-obj_cam_T[2,3,:]):
        print(str(models[i]))
        print(i)
        # copy all  the points
        cloud_temp = copy.deepcopy(all_points[str(models[i])]).transpose().astype(np.float32)
        cloud_temp = np.vstack(( cloud_temp, np.ones((1,cloud_temp.shape[1])) )).astype(np.float32)
        
        # Then transform it into camera's coordinate system
        cloud_cam = np.dot(  obj_cam_T[:, :, i]  , cloud_temp).astype(np.float32)
        
        # transform from camera-link to camera optical link
        cloud_optical = np.dot(op2cam_T, cloud_cam).astype(np.float32)
        
        # perspective projection into ifrom rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicymage-plane
        x,y,z,w = np.dot( cam_P, cloud_optical ).astype(np.float32) #This is the projection step
        print(z)
        x = x / z
        y = y / z
        
        print(x)

        #clips out all the points projected out of image height and width
        clipping = np.logical_and( np.logical_and(x>=0, x<=640) , np.logical_and(y>=0, y<=480) )
        x = x[np.where(clipping)]
        y = y[np.where(clipping)]
        
        #print(np.shape(x))
        
        #Leave the background black
        pixels = np.vstack((x,y)).transpose()
        pixels = np.array(pixels, dtype=np.uint16)
        #print(np.shape(pixels))
        pixels_list.append([pixels])
        
        this_mask = np.zeros((cam_info_msg.height, cam_info_msg.width), dtype= np.uint8)
        
        for point in pixels:
            this_mask[point[1]-1, point[0]-1] = 255
        
        this_mask = cv_im_fill(this_mask)
        
        this_mask[this_mask.nonzero()] = 1.05*np.ceil(255*(i+1)/n_models)
        r,c = this_mask.nonzero()
        #print(np.unique(this_mask[r,c]))
        #mask_list.append(this_mask)
        
        bin_mask[this_mask.nonzero()] = 0
        bin_mask += this_mask
           
    return bin_mask
#endregion

class DataRenderGazebo(Node):
    def __init__(self):
        super().__init__('data_render_gazebo')
        self.get_logger().info('Initializing data rendering gazebo v10')
        self.bridge = CvBridge()
        self.declare_parameters(namespace='', parameters=[
            ('camera_info_topic', '/kinect1/camera/camera_info'),
            ('rgb_topic', '/kinect1/camera/image_raw'),
            ('depth_topic', '/kinect1/camera/depth/image_raw'),
            ('models', ['apple', 'banana']) # for example
        ])
        
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.models_param = self.get_parameter('models').get_parameter_value().string_array_value

        # Initialize services
        #lowest qos
        # Initialize services with lowest QoS

        self.get_model_state_client = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.set_link_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        
        # Wait for services
        while not self.get_model_state_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('[get_entity_state] Service not available, waiting again...')
            time.sleep(0.5)          # if not self.startchecking:
        #     return False# Small delay to prevent spamming in tight loop
        while not self.set_link_state_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('[set_entity_state] Service not available, waiting again...')
            time.sleep(0.5)  # Small delay to prevent spamming in tight loop

  

        # Wait for camera info
        self.get_logger().info('Waiting for camera info...')
        self.cam_info_msg = self.wait_for_message(self.camera_info_topic, CameraInfo)
        self.rgb_topic_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 1)
        self.depth_topic_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 1)


        self.rgb_image_checked = None # for image that has already been saved to compared the update
        self.depth_image_checked = None # for image that has already been saved to compared the update

        # Set up the timer with a callback frequency of 10 second
        #self.timer = self.create_timer(10.0, self.timer_callback)

        # image bridge
        self.bridge = CvBridge()

        # initial setup parameters
        self.sample_num = 0
        self.phi_init = 35
        self.theta_init = 0
        self.dist_init = 0.15

        self.phi = self.phi_init
        self.theta = self.theta_init
        self.dist = self.dist_init

        self.max_phi = 90
        self.max_theta = 360
        self.max_dist = 0.5

        self.phi_increment = 15
        self.theta_increment = 15
        self.dist_increment = 0.125
        self.startchecking = True
        self.model_num = len(self.models_param)

        # setup the enviornment for datageneration
        self.setup()
        self.get_logger().info('Node preparation done, start to generate synthetic data')

        self.data_loop()

    def setup(self):
        self.get_logger().info('Setting up the environment')

        # convert the meshes to pointclouds 
        self.get_logger().info('Generating the pointclouds for all models. hold on , This could take few minutes ...')
        self.all_points, self.all_pclds = self.mesh2pcld()
        
        self.rgb_image = None
        self.depth_image = None
        # self.get_logger().info('Setting first camera pose')
        # camPos,camTwist, X, Y,Z, cam_euler  = calc_params(self.phi, self.theta, self.dist)
        # self.currentCameraSetting = (camPos, camTwist, X, Y,Z, cam_euler)
        # self.set_cam_state_gazebo(camPos,camTwist)

        self.grid_based_sampling = True

        # prepare the random phi dist and theta pairs 
        if not self.grid_based_sampling:
            self.random_samples = [
                (
                    random.uniform(self.phi_init, self.max_phi),
                    random.uniform(self.dist_init, self.max_dist),
                    random.uniform(self.theta_init, self.max_theta)
                ) for _ in range(10000)
            ]


    def data_loop(self):
        while True:
            # set the camera
            # Set camera pose in gazebo for next trial
            self.get_logger().info(f'======= Generating sample {self.sample_num}, phi = {self.phi}, theta = {self.theta}, dist = {self.dist} =======')

            camPos,camTwist, X, Y,Z, cam_euler  = calc_params(self.phi, self.theta, self.dist)
            self.set_cam_state_gazebo(camPos=camPos, camTwist=camTwist)
            self.currentCameraSetting = (camPos, camTwist, X, Y,Z, cam_euler)

            self.get_logger().info('wait some time after camera being set')
            time.sleep(0.05)
            #check image
            while not self.check_new_image():
                self.get_logger().info('waiting for new image, reset the camera')
                self.set_cam_state_gazebo(camPos=camPos, camTwist=camTwist)
                time.sleep(0.05)

            self.process_image()
            # save the image 
            cv_depthImage = convert_types(self.depth_image_checked,0,3, 0,65535, np.uint16) ## 0 - 3m is the input range of kinect depth
            self.get_logger().info(f'Writing Images for sample number {self.sample_num}')
            cv2.imwrite('dataset/rgb/'+str(self.sample_num)+'.png', self.rgb_image_checked)
            cv2.imwrite('dataset/depth/'+str(self.sample_num)+'.png',cv_depthImage)
                
            try:
                resp = self.get_object_states()
            except Exception as inst:
                print('Error in gazebo/get_link_state service request: ' + str(inst) )
                
            cam_world_T = get_camera_extrinsics(self.phi,self.theta, self.dist)
            gt_dict,obj_cam_T = get_object2cam_pose(resp, self.model_num ,cam_world_T)
            self.get_logger().info('Saving the fucking meta')

            sio.savemat('dataset/meta/'+str(self.sample_num)+'-meta.mat',gt_dict)

            self.get_logger().info('Saving the fucking mask')
            bin_mask= pcl_2_binary_mask(obj_cam_T, self.model_num, self.all_points, self.cam_info_msg, self.models_param)
            
            cv2.imwrite('dataset/mask/'+str(self.sample_num)+'.png',bin_mask)
            

            self.get_logger().info('Updating the camera pose')
            # Update angles and distance for the next position
            if self.set_next_camPos():
                break

            self.sample_num += 1

        


        
    
    #%% Load the meshes of all objects convert them to point clouds
    # combine and return the pointclouds of all meshes in a dictionary
    def mesh2pcld(self):
        all_points = {}
        all_pclds  = {}

        for i in range (0,self.model_num):
            mesh = o3d.io.read_triangle_mesh(model_workfolder+str(self.models_param[i])+'/meshes/'+str(self.models_param[i])+'.stl')
            poisson_pcld = mesh.sample_points_poisson_disk(number_of_points=30000) 
            all_pclds[str(self.models_param[i])] = poisson_pcld
            o3d.io.write_point_cloud('dataset/model_pointcloud/'+str(self.models_param[i])+'.ply', poisson_pcld )
            all_points[str(self.models_param[i])] = np.asarray(poisson_pcld.points)#, dtype= np.float32)   
        return all_points, all_pclds
    
    # to check new image and process accordingly
    def timer_callback(self):
        # Check if the new image is available and process accordingly
        if self.check_new_image():
            # record the new image and do possible manipulation
            self.process_image()
            self.get_logger().info(f'======= Generating sample {self.sample_num}, phi = {self.phi}, theta = {self.theta}, dist = {self.dist} =======')
            # save the image 
            cv_depthImage = convert_types(self.depth_image_checked,0,3, 0,65535, np.uint16) ## 0 - 3m is the input range of kinect depth
            self.get_logger().info(f'Writing Images for sample number {self.sample_num}')
            cv2.imwrite('dataset/rgb/'+str(self.sample_num)+'.png', self.rgb_image_checked)
            cv2.imwrite('dataset/depth/'+str(self.sample_num)+'.png',cv_depthImage)
            
            # get the previous camera state
            (prev_camPos, prev_camTwist, prev_X, prev_Y,prev_Z, prev_cam_euler) = self.currentCameraSetting


            try:
                resp = self.get_object_states()
            except Exception as inst:
                     print('Error in gazebo/get_link_state service request: ' + str(inst) )
                
            cam_world_T = get_camera_extrinsics(self.phi,self.theta, self.dist)
            gt_dict,obj_cam_T = get_object2cam_pose(resp, self.model_num ,cam_world_T)
            self.get_logger().info('Saving the fucking meta')

            sio.savemat('dataset/meta/'+str(self.sample_num)+'-meta.mat',gt_dict)

            self.get_logger().info('Saving the fucking mask')
            bin_mask= pcl_2_binary_mask(obj_cam_T, self.model_num, self.all_points, self.cam_info_msg, self.models_param)
            
            cv2.imwrite('dataset/mask/'+str(self.sample_num)+'.png',bin_mask)
            

            self.get_logger().info('Updating the camera pose')
            # Update angles and distance for the next position
            self.set_next_camPos()


            # Set camera pose in gazebo for next trial
            camPos,camTwist, X, Y,Z, cam_euler  = calc_params(self.phi, self.theta, self.dist)
            self.set_cam_state_gazebo(camPos=camPos, camTwist=camTwist)
            self.currentCameraSetting = (camPos, camTwist, X, Y,Z, cam_euler)

            self.startchecking = True # can start to check for new image

            self.sample_num += 1
            self.get_logger().info(f'======= Generated sample {self.sample_num}=======')

        else:
            #Try setting state again. Sometimes gazebo trips out as well.
            self.set_cam_state_gazebo(camPos=self.currentCameraSetting[0], camTwist=self.currentCameraSetting[1])
        
    def get_object_states(self):

        resp = []
        # Request model states
        for model_name in self.models_param:
            request = GetEntityState.Request()
            #print(request)
            request.name = 'apple'
            request.reference_frame = 'world'
            self.get_logger().info(f'Sending request of {request} for model {model_name}')
            try:
                future = self.get_model_state_client.call_async(request)
                self.get_logger().info('Waiting Service...')
                rclpy.spin_until_future_complete(self, future)
                response = future.result()
                self.get_logger().info('done')
                if response.success:
                    resp.append(response.state)
                    self.get_logger().info(f'State for model {model_name}: {response.state}')
                else:
                    self.get_logger().info(f'Failed to get state for model {model_name}')
            except Exception as e:
                self.get_logger().error(f'Error in gazebo/get_model_state service request: {str(e)}')

        return resp
    



    def set_cam_state_gazebo(self, camPos, camTwist):


        # Create the service message
        cam_state = EntityState()
        cam_state.name = 'kinect_ros::link'  # Ensure the entity name matches your simulation specifics
        cam_state.pose = camPos
        cam_state.twist = camTwist
        cam_state.reference_frame = 'world'  # Setting relative to world frame

        # Log the action
        self.get_logger().info(f'Transforming camera to pose: {self.sample_num}')

        # Create the service request
        request = SetEntityState.Request()
        request.state = cam_state

        # Call the service asynchronously
        try:
            send_request_future = self.set_link_state_client.call_async(request)
            # Wait for the future to complete
            rclpy.spin_until_future_complete(self, send_request_future)
            response = send_request_future.result()
            if response.success:
                self.get_logger().info('Camera state successfully updated.')
            else:
                self.get_logger().info('Failed to update camera state.')
        except Exception as inst:
            self.get_logger().error('Error in gazebo/set_entity_state service request: ' + str(inst))

    
    def process_image(self):
        self.rgb_image_checked = self.rgb_image
        self.depth_image_checked = self.depth_image

    def check_new_image(self):
        # if not self.startchecking:
        #     return False
        self.get_logger().info('checking new image')
        
        if(self.rgb_image is None or self.depth_image is None):
            self.get_logger().info('rgb or depth image is not ready')
            return False

        if(self.rgb_image_checked is None or self.depth_image_checked is None):
            self.get_logger().info('First image checked')
            return True
        
        # check for mean rgb

        rgb_duplicate = abs(np.mean(self.rgb_image - self.rgb_image_checked)) > 0.5

        # check for mean depth
        depth_duplicate = abs(np.nanmean(self.depth_image - self.depth_image_checked))> 0.5
        self.get_logger().info(f'rgb score {abs(np.mean(self.rgb_image - self.rgb_image_checked))}, depth score {abs(np.nanmean(self.depth_image - self.depth_image_checked))}')
        return (rgb_duplicate or depth_duplicate)
    
    def wait_for_message(self, topic_name, msg_type):
            
        future = Future()
        
        def callback(msg):
            if not future.done():
                future.set_result(msg)
        
        self.create_subscription(msg_type, topic_name, callback, 10)
        
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return future.result()
    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        self.rgb_image = cv_image

    def depth_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        self.depth_image = cv_image

    def set_next_camPos(self):  
        # go for grid based sampling
        if self.grid_based_sampling:
            self.theta += self.theta_increment # can change these offset
            if self.theta >= self.max_theta:
                self.theta = self.theta_init
                self.phi += self.phi_increment # can change these offset
            if self.phi >= self.max_phi:
                self.phi = self.phi_init
                self.dist += self.dist_increment # can change these offset
            if self.dist > self.max_dist:
                self.get_logger().info('Completed dataset generation')
                self.destroy_timer(self.timer)  # Stop the timer if done
                return True

            return False
        # go for random sampling
        else:
            if self.sample_num >= len(self.random_samples):
                self.get_logger().info('Completed dataset generation')
                self.destroy_timer(self.timer)  # Stop the timer if done
                return True

            self.phi, self.dist, self.theta = self.random_samples[self.sample_num]
            self.sample_num += 1
            return False



def main(args=None):

    #%% Create the directories if they don't already exist
    if not os.path.exists('dataset/rgb'):
        os.makedirs('dataset/rgb')
    if not os.path.exists('dataset/depth'):
        os.makedirs('dataset/depth')
    if not os.path.exists('dataset/meta'):
        os.makedirs('dataset/meta')
    if not os.path.exists('dataset/mask'):
        os.makedirs('dataset/mask')
    if not os.path.exists('dataset/model_pointcloud'):
        os.makedirs('dataset/model_pointcloud') 

    rclpy.init(args=args)
    data_render_node = DataRenderGazebo()
    try:
        rclpy.spin(data_render_node)
    except KeyboardInterrupt:
        pass
    finally:
        data_render_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
