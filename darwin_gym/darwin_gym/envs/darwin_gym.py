
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import os, inspect
import pybullet_data

from pybullet_utils import bullet_client

from pkg_resources import parse_version

try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except:
  pass

class URDFBulletEnv(gym.Env):
  """
	Base  for Bullet physics simulation loading urdf .urdf environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

  def __init__(self, robot, render=False):
    self.scene = None
    self.physicsClientId = -1
    self.ownsPhysicsClient = 0
    self.camera = Camera(self)
    self.isRender = render
    self.robot = robot
    self.seed()
    self._cam_dist = 3
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._render_width = 320
    self._render_height = 240

    self.action_space = robot.action_space
    self.observation_space = robot.observation_space
    #self.reset()

  def configure(self, args):
    self.robot.args = args

  def seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    return [seed]

  def reset(self):
    if (self.physicsClientId < 0):
      self.ownsPhysicsClient = True

      if self.isRender:
        print("sssssssssssssssssssssssssssssssssssssssssssss")
        self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      else:
        self._p = bullet_client.BulletClient()
      self._p.resetSimulation()
      self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
      #optionally enable EGL for faster headless rendering
      try:
        if os.environ["PYBULLET_EGL"]:
          con_mode = self._p.getConnectionInfo()['connectionMethod']
          if con_mode==self._p.DIRECT:
            egl = pkgutil.get_loader('eglRenderer')
            if (egl):
              self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
              self._p.loadPlugin("eglRendererPlugin")
      except:
        pass
      
      self.physicsClientId = self._p._client
      self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    if self.scene is None:
      self.scene = self.create_single_player_scene(self._p)
    if not self.scene.multiplayer and self.ownsPhysicsClient:
      self.scene.episode_restart(self._p)

    self.robot.scene = self.scene

    self.frame = 0
    self.done = 0
    self.reward = 0
    dump = 0
    s = self.robot.reset(self._p)
    self.potential = self.robot.calc_potential()
    return s

  def camera_adjust(self):
    pass

  def render(self, mode='human', close=False):
  
    if mode == "human":
      self.isRender = True
    if self.physicsClientId>=0:
      self.camera_adjust()

    if mode != "rgb_array":
      return np.array([])

    base_pos = [0, 0, 0]
    if (hasattr(self, 'robot')):
      if (hasattr(self.robot, 'body_real_xyz')):
        base_pos = self.robot.body_real_xyz
    if (self.physicsClientId>=0):
      view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
      proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(self._render_width) /
                                                     self._render_height,
                                                     nearVal=0.01,
                                                     farVal=100.0)
      (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                              height=self._render_height,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

      self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    else:
      px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def close(self):
    if (self.ownsPhysicsClient):
      if (self.physicsClientId >= 0):
        self._p.disconnect()
    self.physicsClientId = -1

  def HUD(self, state, a, done):
    pass

  # def step(self, *args, **kwargs):
  # 	if self.isRender:
  # 		base_pos=[0,0,0]
  # 		if (hasattr(self,'robot')):
  # 			if (hasattr(self.robot,'body_xyz')):
  # 				base_pos = self.robot.body_xyz
  # 				# Keep the previous orientation of the camera set by the user.
  # 				#[yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
  # 				self._p.resetDebugVisualizerCamera(3,0,0, base_pos)
  #
  #
  # 	return self.step(*args, **kwargs)
  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed


class Camera:

  def __init__(self, env):
    self.env = env
    pass

  def move_and_look_at(self, i, j, k, x, y, z):
    lookat = [x, y, z]
    camInfo = self.env._p.getDebugVisualizerCamera()
    
    distance = camInfo[10]
    pitch = camInfo[9]
    yaw = camInfo[8]
    self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)





class Scene:
  "A base class for single- and multiplayer scenes"

  def __init__(self, bullet_client, gravity, timestep, frame_skip):
    self._p = bullet_client
    self.np_random, seed = gym.utils.seeding.np_random(None)
    self.timestep = timestep
    self.frame_skip = frame_skip

    self.dt = self.timestep * self.frame_skip
    self.cpp_world = World(self._p, gravity, timestep, frame_skip)

    self.test_window_still_open = True  # or never opened
    self.human_render_detected = False  # if user wants render("human"), we open test window

    self.multiplayer_robots = {}

  def test_window(self):
    "Call this function every frame, to see what's going on. Not necessary in learning."
    self.human_render_detected = True
    return self.test_window_still_open

  def actor_introduce(self, robot):
    "Usually after scene reset"
    if not self.multiplayer: return
    self.multiplayer_robots[robot.player_n] = robot

  def actor_is_active(self, robot):
    """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
    return not self.multiplayer

  def episode_restart(self, bullet_client):
    "This function gets overridden by specific scene, to reset specific objects into their start positions"
    self.cpp_world.clean_everything()
    #self.cpp_world.test_window_history_reset()

  def global_step(self):
    """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
    self.cpp_world.step(self.frame_skip)


class SingleRobotEmptyScene(Scene):
  multiplayer = False  # this class is used "as is" for InvertedPendulum, Reacher


class World:

  def __init__(self, bullet_client, gravity, timestep, frame_skip):
    self._p = bullet_client
    self.gravity = gravity
    self.timestep = timestep
    self.frame_skip = frame_skip
    self.numSolverIterations = 500
    self.clean_everything()

  def clean_everything(self):
    #p.resetSimulation()
    self._p.setGravity(0, 0, -self.gravity)

    #self._p.setRealTimeSimulation(1)
    self._p.setDefaultContactERP(0.9)
    #print("self.numSolverIterations=",self.numSolverIterations)
    self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep * self.frame_skip,
                                      numSolverIterations=self.numSolverIterations,
                                      numSubSteps=self.frame_skip)

  def step(self, frame_skip):
    self._p.stepSimulation()


class StadiumScene(Scene):
  zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
  stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
  stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
  stadiumLoaded = 0

  def episode_restart(self, bullet_client):
    self._p = bullet_client
    Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
    if (self.stadiumLoaded == 0):
      self.stadiumLoaded = 1

      # stadium_pose = cpp_household.Pose()
      # if self.zero_at_running_strip_start_line:
      #	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

      #filename = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
      #self.ground_plane = self._p.loadURDF(filename)
      #filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
      #self.ground_plane_mjcf = self._p.loadSDF(filename)
      #
      
        #self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
      #self._p.changeVisualShape(self.ground_plane, -1, rgbaColor=[1, 1, 1, 0.8])
      #self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,self.ground_plane)

      filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
      self.ground_plane = self._p.loadSDF(filename)
      #filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
      #self.ground_plane_mjcf = self._p.loadSDF(filename)
      #
      for i in self.ground_plane:
        self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.9)
        self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,i)

      #	for j in range(p.getNumJoints(i)):
      #		self._p.changeDynamics(i,j,lateralFriction=0)
      #despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground


class SinglePlayerStadiumScene(StadiumScene):
  "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
  multiplayer = False


class MultiplayerStadiumScene(StadiumScene):
  multiplayer = True
  players_count = 3

  def actor_introduce(self, robot):
    StadiumScene.actor_introduce(self, robot)
    i = robot.player_n - 1  # 0 1 2 => -1 0 +1
    robot.move_robot(0, i, 0)


class XmlBasedRobot:
  """
	Base class for mujoco .xml based agents.
	"""

  self_collision = True

  def __init__(self, robot_name, action_dim, obs_dim, self_collision):
    self.parts = None
    self.objects = []
    self.jdict = None
    self.ordered_joints = None
    self.robot_body = None

    high = np.ones([action_dim]) * 0.01
    self.action_space = gym.spaces.Box(-high, high)
    high = np.inf * np.ones([obs_dim])
    self.observation_space = gym.spaces.Box(-high, high)

    #self.model_xml = model_xml
    self.robot_name = robot_name
    self.self_collision = self_collision

  def addToScene(self, bullet_client, bodies):
    self._p = bullet_client

    if self.parts is not None:
      parts = self.parts
    else:
      parts = {}

    if self.jdict is not None:
      joints = self.jdict
    else:
      joints = {}

    if self.ordered_joints is not None:
      ordered_joints = self.ordered_joints
    else:
      ordered_joints = []

    if np.isscalar(bodies):  # streamline the case where bodies is actually just one body
      bodies = [bodies]

    dump = 0
    for i in range(len(bodies)):
      if self._p.getNumJoints(bodies[i]) == 0:
        part_name, robot_name = self._p.getBodyInfo(bodies[i])
        self.robot_name = robot_name.decode("utf8")
        part_name = part_name.decode("utf8")
        parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
        #print("khdfkjsahi",part_name)
      for j in range(self._p.getNumJoints(bodies[i])):
        jointInfo = self._p.getJointInfo(bodies[i], j)
        MaxForce=jointInfo[10]/150.0
        MaxVelocity=jointInfo[11]/100.0
        self._p.setJointMotorControl2(bodies[i],
                                      j,
                                      pybullet.POSITION_CONTROL,
                                      targetPosition=0,
                                      force=0)
        
        joint_name = jointInfo[1]
        part_name = jointInfo[12]

        joint_name = joint_name.decode("utf8")
        part_name = part_name.decode("utf8")
        #print(i)
        if dump: print("ROBOT PART '%s'" % part_name)
        if dump:
          print(
              "ROBOT JOINT '%s'" % joint_name
          )  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

        parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

        if part_name == self.robot_name:
          self.robot_body = parts[part_name]
          print("parttttttttttttttt",part_name)

        if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
          parts[self.robot_name] = BodyPart(self._p, self.robot_name, bodies, 0, -1)
          self.robot_body = parts[self.robot_name]
          #print("parttttttttttttttt")

        if joint_name[:6] == "ignore" or joint_name[:8] == "jointfix" :
          Joint(self._p, joint_name, bodies, i, j).fix_motor()
          continue

        if joint_name[:8] != "jointfix" or joint_name[:6] != "ignore":
          joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
          ordered_joints.append(joints[joint_name])

          joints[joint_name].power_coef = 100.0

        # TODO: Maybe we need this
        # joints[joint_name].power_coef, joints[joint_name].max_velocity = joints[joint_name].limits()[2:4]
        # self.ordered_joints.append(joints[joint_name])
        # self.jdict[joint_name] = joints[joint_name]

    return parts, joints, ordered_joints, self.robot_body

  def reset_pose(self, position, orientation):
    self.parts[self.robot_name].reset_pose(position, orientation)

class URDFBasedRobot(XmlBasedRobot):
  """
	Base class for URDF .xml based robots.
	"""

  def __init__(self,
               model_urdf,
               robot_name,
               action_dim,
               obs_dim,
               basePosition=[0, 0, 0.4],
               baseOrientation=[0, 0, 0, 1],
               fixed_base=False,
               self_collision=True):
    XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)

    self.model_urdf = model_urdf
    #print(self.model_urdf)
    self.basePosition = basePosition
    self.baseOrientation = baseOrientation
    self.fixed_base = fixed_base
    self.doneLoading=0

  def reset(self, bullet_client):
    self._p = bullet_client
    

    #print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))
    if (self.doneLoading == 0):
      self.ordered_joints = []
      self.doneLoading=1
      #get_cube(self._p,12,0,0)
      if self.self_collision:
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
            self._p,
            self._p.loadURDF(self.model_urdf,
                            basePosition=self.basePosition,
                            baseOrientation=self.baseOrientation,
                            useFixedBase=self.fixed_base)) 
      else:
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
            self._p,
            self._p.loadURDF( self.model_urdf,
                            basePosition=self.basePosition,
                            baseOrientation=self.baseOrientation,
                            useFixedBase=self.fixed_base))

    self.robot_specific_reset(self._p)

    s = self.calc_state(
    )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
    self.potential = self.calc_potential()

    return s

  def calc_potential(self):
    return 0

class Pose_Helper:  # dummy class to comply to original interface

  def __init__(self, body_part):
    self.body_part = body_part

  def xyz(self):
    return self.body_part.current_position()

  def rpy(self):
    return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

  def orientation(self):
    return self.body_part.current_orientation()

class BodyPart:

  def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
    self.bodies = bodies
    self._p = bullet_client
    self.bodyIndex = bodyIndex
    self.bodyPartIndex = bodyPartIndex
    self.body_name=body_name
    self.initialPosition = self.current_position()
    self.initialOrientation = self.current_orientation()
    self.bp_pose = Pose_Helper(self)

  def state_fields_of_pose_of(
      self, body_id,
      link_id=-1):  # a method you will most probably need a lot to get pose and orientation
    if link_id == -1:
      (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
    else:
      (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
    return np.array([x, y, z, a, b, c, d])

  def get_position(self):
    return self.current_position()

  def get_pose(self):
    return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

  def speed(self):
    if self.bodyPartIndex == -1:
      (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
    else:
      (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
          self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
    return np.array([vx, vy, vz])

  def current_position(self):
    return self.get_pose()[:3]

  def current_orientation(self):
    return self.get_pose()[3:]

  def get_orientation(self):
    return self.current_orientation()

  def reset_position(self, position):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position,
                                            self.get_orientation())

  def reset_orientation(self, orientation):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(),
                                            orientation)

  def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
    self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

  def reset_pose(self, position, orientation):
    self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

  def pose(self):
    return self.bp_pose

  def contact_list(self):
    return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)

class Joint:

  def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
    self.bodies = bodies
    self._p = bullet_client
    self.bodyIndex = bodyIndex
    self.jointIndex = jointIndex
    self.joint_name = joint_name

    jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
    self.lowerLimit = jointInfo[8]
    self.upperLimit = jointInfo[9]
    self.MaxForce=jointInfo[10]*3#/15.0
    self.MaxVelocity=jointInfo[11]#/10.0
    #print("joint ",self.joint_name,"vel ",self.MaxVelocity, self.MaxForce )
    self.power_coeff = 0

  def set_state(self, x, vx):
    self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

  def current_position(self):  # just some synonyme method
    return self.get_state()

  def current_relative_position(self):
    pos, vel,_ = self.get_state()
    pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
    return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), 0.1 * vel)

  def get_state(self):
    x, vx, _, tx = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
    return x, vx, tx

  def get_position(self):
    x, _,_ = self.get_state()
    return x

  def get_orientation(self):
    _, r ,_= self.get_state()
    return r

  def get_velocity(self):
    _, vx,_ = self.get_state()
    return vx
  def get_torque(self):
    _,_,tx=self.get_state()
    return tx 

  def set_position(self, position):
    #print("bc")
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  pybullet.POSITION_CONTROL,
                                  targetPosition=position,
                                  targetVelocity=0,
                                  positionGain=0.6,
                                  velocityGain=0.5,
                                  force=self.MaxForce)
                                  #force=self.MaxForce)
                                

  def set_velocity(self, velocity):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  pybullet.VELOCITY_CONTROL,
                                  targetVelocity=velocity)

  def set_motor_torque(self, torque):  # just some synonyme method
    self.set_torque(torque)

  def set_torque(self, torque):
    self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex],
                                  jointIndex=self.jointIndex,
                                  controlMode=pybullet.TORQUE_CONTROL,
                                  force=torque)  #, positionGain=0.1, velocityGain=0.1)



  def reset_current_position(self, position, velocity):  # just some synonyme method
    self.reset_position(position, velocity)

  def reset_position(self, position, velocity):
    self._p.resetJointState(self.bodies[self.bodyIndex],
                            self.jointIndex,
                            targetValue=position,
                            targetVelocity=velocity)
    self.disable_motor()

  def disable_motor(self):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  controlMode=pybullet.POSITION_CONTROL,
                                  targetPosition=0,
                                  targetVelocity=0,
                                  positionGain=0.1,
                                  velocityGain=0.1,
                                  force=0)


  def fix_motor(self):
    self._p.setJointMotorControl2(self.bodies[self.bodyIndex],
                                  self.jointIndex,
                                  controlMode=pybullet.POSITION_CONTROL,
                                  targetPosition=0,
                                  targetVelocity=0,
                                  positionGain=0.5,
                                  velocityGain=0.5,
                                  force=100)

class WalkerBase(URDFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    URDFBasedRobot.__init__(self,fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    #get_cube(self._p,2,0,0)
    for j in self.ordered_joints:
      j.reset_current_position(0,0)#self.np_random.uniform(low=-0.001, high=0.001), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    #print("ssssssssssssss")
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    #for i in self.ordered_joints:
    #  print(" a",i.joint_name," ",i.jointIndex)

    self.joint_speeds = j[1::2]
    #print(self.joint_speeds)
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt

class Humanoid(WalkerBase):
  self_collision = True
  foot_list = ["MP_ANKLE2_L", "MP_ANKLE2_R"]  # "left_hand", "right_hand"

  def __init__(self):
    WalkerBase.__init__(self,
                        "/content/DarwinOp_gym/darwin_gym/darwin_gym/envs/darwin4.urdf",
                        'MP_BODY',
                        action_dim=14,
                        obs_dim=38,
                        power=0.06)
    # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    self.motor_names = ["j_shoulder_l", "j_high_arm_l", "j_low_arm_l"]
    self.motor_power = [100, 100, 100]
    self.motor_names += ["j_shoulder_r", "j_high_arm_r", "j_low_arm_r"]
    self.motor_power += [100, 100, 100]
    self.motor_names += ["j_pelvis_l", "j_thigh1_l", "j_thigh2_l", "j_tibia_l"]#,"j_ankle1_l","j_ankle2_l"]
    self.motor_power += [100, 100, 100, 100]#, 100, 100]
    self.motor_names +=["j_pelvis_r", "j_thigh1_r", "j_thigh2_r", "j_tibia_r"]#,"j_ankle1_r","j_ankle2_r"]
    self.motor_power += [100, 100, 100,100]#, 100,100]
    
    
    self.motors = [self.jdict[n] for n in self.motor_names]
    """
    if self.random_yaw:
      position = [0, 0, 0]
      orientation = [0, 0, 0]
      yaw = self.np_random.uniform(low=-3.14, high=3.14)
      if self.random_lean and self.np_random.randint(2) == 0:
        cpose.set_xyz(0, 0, 1.4)
        if self.np_random.randint(2) == 0:
          pitch = np.pi / 2
          position = [0, 0, 0.45]
        else:
          pitch = np.pi * 3 / 2
          position = [0, 0, 0.25]
        roll = 0
        orientation = [roll, pitch, yaw]
      else:
        position = [0, 0, 1.4]
        orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
      self.robot_body.reset_position(position)
      self.robot_body.reset_orientation(orientation)
    """  
    self.initial_z = 0.342

  random_yaw = False
  random_lean = False

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    force_gain = 1
    #val=a*0+0.1
    for i, m, power in zip(range(14), self.motors, self.motor_power):
      #m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -.1, +.1)))
      debug_torque=0
      if debug_torque and i==13:
        print("position_intial",m.get_position())

    
      m.set_position(np.clip(m.get_position()+self.power*np.clip(a[i], -1, +1),m.lowerLimit,m.upperLimit)) #np.clip(a[i], -.01, +.01))#np.clip(m.get_position(),m.lowerLimit,m.upperLimit))  
      
      if debug_torque and abs(m.get_position()-(np.clip(m.get_position()+np.clip(a[i], -.01, +.01),m.lowerLimit,m.upperLimit)))>0.001 and i==13:
        print("****888*******888888888888888888888888*************")
        print("forced", m.joint_name," :",m.get_torque(),self.power*np.clip(a[i], -1, +1))
        #print(a[i]," input ") 
        
        
        print("velocity",m.get_orientation())
        print("position",m.get_position(),np.clip(m.get_position()+self.power*np.clip(a[i], -1, +1),m.lowerLimit,m.upperLimit))
      

  def alive_bonus(self, z, pitch):
    return +2 if z > 0.2 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


def get_cube(_p, x, y, z):
  body = _p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"), [x, y, z])
  _p.changeDynamics(body, -1, mass=1.2)  #match Roboschool
  part_name, _ = _p.getBodyInfo(body)
  part_name = part_name.decode("utf8")
  bodies = [body]
  return BodyPart(_p, part_name, bodies, 0, -1)


def get_sphere(_p, x, y, z):
  body = _p.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere2red_nocol.urdf"), [x, y, z])
  part_name, _ = _p.getBodyInfo(body)
  part_name = part_name.decode("utf8")
  bodies = [body]
  return BodyPart(_p, part_name, bodies, 0, -1)

class WalkerBaseBulletEnv(URDFBulletEnv):

  def __init__(self, robot, render=True):
    # print("WalkerBase::__init__ start")
    self.camera_x = 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.stateId = -1

    URDFBulletEnv.__init__(self, robot, render)


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 1,
                                                  frame_skip=1)
    return self.stadium_scene
  def reset(self):
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r = URDFBulletEnv.reset(self)
    
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane)
    #self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
    #                        self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    #print(self.ground_ids )

    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)

    return r

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    "Used by multiplayer stadium to move sideways, to another running lane."
    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(
        init_x, init_y, init_z
    )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  #foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = 2.6*float(self.potential - potential_old)

    feet_collision_cost = 0.0
    
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print(contact_ids)
      #print(f.body_name)
      #print("i",i)
        
      #print("contacts:", contact_ids)
      #for x in f.contact_list():
      #  print(x)
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0
    #print("speed",self.robot.joint_speeds)
    #print("postion",a)
    electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
    ))  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    debugmode = 0
    if (debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress , potential_old,self.potential)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [
        self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
    ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), {}

  def camera_adjust(self):
    x, y, z = self.robot.body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 0.1, x, y,0.1)

class DarwinBulletEnv(WalkerBaseBulletEnv):
  
  def __init__(self, robot=Humanoid(), render=False):
    self.robot = robot
    WalkerBaseBulletEnv.__init__(self, self.robot, render)
    self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
    self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost





