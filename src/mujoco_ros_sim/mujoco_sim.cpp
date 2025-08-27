#include "mujoco_ros_sim/mujoco_sim.hpp"
#include "mujoco_ros_sim/utils.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/qos.hpp>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std::chrono_literals;

namespace {
// 샘플 코드와 동일한 네임스페이스 별칭
namespace mj  = ::mujoco;
namespace mju = ::mujoco::sample_util;

// 전역 m, d (샘플 구조를 그대로 사용)
mjModel* g_model = nullptr;
mjData*  g_data  = nullptr;

// 플러그인 디렉토리 스캔(샘플과 동일)
void scanPluginLibraries();

// 파일 로드(샘플과 동일 로직)
mjModel* LoadModel(const char* file, mj::Simulate& sim);

// 물리 루프(샘플과 동일)
void PhysicsLoop(mj::Simulate& sim);

// 물리 스레드 엔트리(샘플)
void PhysicsThread(mj::Simulate* sim, const char* filename) {
  if (filename) {
    sim->LoadMessage(filename);
    g_model = LoadModel(filename, *sim);
    if (g_model) {
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
      g_data = mj_makeData(g_model);
    }
    if (g_data) {
      sim->Load(g_model, g_data, filename);
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
      mj_forward(g_model, g_data);
    } else {
      sim->LoadMessageClear();
    }
  }
  PhysicsLoop(*sim);
  mj_deleteData(g_data); g_data = nullptr;
  mj_deleteModel(g_model); g_model = nullptr;

}
  std::atomic<mujoco::Simulate*> g_sim_for_shutdown{nullptr};
} // anonymous


namespace mujoco_ros_sim {

// ---------- helpers ----------
builtin_interfaces::msg::Time MujocoSimNode::toTimeMsg(double tsec) {
  builtin_interfaces::msg::Time t;
  const auto s = static_cast<int32_t>(std::floor(tsec));
  const auto n = static_cast<uint32_t>(std::llround((tsec - static_cast<double>(s)) * 1e9));
  t.sec = s; t.nanosec = n;
  return t;
}

// ---------- ctor/dtor ----------
MujocoSimNode::MujocoSimNode()
: rclcpp::Node("mujoco_sim_node")
{
  // parameters
  this->declare_parameter<std::string>("robot_name", "");
  this->declare_parameter<std::string>("model_xml", "");
  this->declare_parameter<bool>("enable_viewer", true);
  this->declare_parameter<double>("camera_fps", 60.0);

  robot_name_    = this->get_parameter("robot_name").as_string();
  model_xml_     = this->get_parameter("model_xml").as_string();
  enable_viewer_ = this->get_parameter("enable_viewer").as_bool();
  camera_fps_    = this->get_parameter("camera_fps").as_double();
  if (camera_fps_ <= 0) camera_fps_ = 60.0;

  // pubs/subs
  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  qos.durability(rclcpp::DurabilityPolicy::Volatile);

  pub_joint_dict_  = this->create_publisher<mujoco_ros_sim_msgs::msg::JointDict>("mujoco_ros_sim/joint_dict", qos);
  pub_sensor_dict_ = this->create_publisher<mujoco_ros_sim_msgs::msg::SensorDict>("mujoco_ros_sim/sensor_dict", qos);
  pub_image_dict_  = this->create_publisher<mujoco_ros_sim_msgs::msg::ImageDict>("mujoco_ros_sim/image_dict", qos);
  pub_joint_state_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
  sub_ctrl_        = this->create_subscription<mujoco_ros_sim_msgs::msg::CtrlDict>(
                        "mujoco_ros_sim/ctrl_dict", qos,
                        std::bind(&MujocoSimNode::onCtrlMsg, this, std::placeholders::_1));

  // UI 준비
  mjv_defaultCamera(&cam_);
  mjv_defaultOption(&opt_);
  mjv_defaultPerturb(&pert_);

  // 플러그인 로드(샘플)
  scanPluginLibraries();

  // 모델 경로 해결 후 UI 시작
  const std::string xml = resolveModelPath();
  if (xml.empty()) {
    RCLCPP_FATAL(get_logger(), "Model XML not found (robot_name or model_xml).");
    throw std::runtime_error("model not found");
  }
  startSimUI(xml);

  // model/data 대기
  {
    auto t0 = std::chrono::steady_clock::now();
    while (rclcpp::ok()) {
      {
        const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
        if (g_model && g_data) { model_ = g_model; data_ = g_data; break; }
      }
      if (std::chrono::steady_clock::now() - t0 > 3s) break;
      std::this_thread::sleep_for(10ms);
    }
  }
  if (!model_ || !data_) {
    RCLCPP_FATAL(get_logger(), "Model/Data not initialized by physics thread.");
    throw std::runtime_error("no model/data");
  }

  RCLCPP_INFO(get_logger(), "%s%s%s", cblue_, print_table(model_xml_, model_).c_str(), creset_);
  

  // dt/딕셔너리
  {
    const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
    dt_ = model_->opt.timestep;
  }
  if (dt_ <= 0.0) dt_ = 0.002;

  buildDictionaries();
  prepareMsgsOnce();
  buildImageSlices();

  // 퍼블리시 스레드 시작
  pub_run_ = true;
  pub_thread_ = std::thread(&MujocoSimNode::publishLoop1k, this);

  // 타이머(퍼블리시: joint_state 전용)
  cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  timer_joint_state_ = this->create_wall_timer(
      10ms, std::bind(&MujocoSimNode::publishJointState, this), cb_group_);

  // 이미지 캡처는 UI 스왑 직후 콜백
  if (enable_viewer_ && sim_) {
    sim_->SetPostRenderCallback([this]() {
      this->captureCamerasOnMainIfDue();
    });
  }

  RCLCPP_INFO(get_logger(), "MuJoCo UI started with model: %s (dt=%.6f)", model_xml_.c_str(), dt_);
}

MujocoSimNode::~MujocoSimNode() {
  stopSimUI();
  pub_run_ = false;
  if (pub_thread_.joinable()) pub_thread_.join();
}

// 퍼블리시 루프
void MujocoSimNode::publishLoop1k() {
  using clock = std::chrono::steady_clock;
  const auto period =
      std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(dt_));
  auto next = clock::now();

  while (pub_run_.load(std::memory_order_relaxed)) {
    next += period;

    // 스냅샷
    {
      const std::unique_lock<std::recursive_mutex> lk(sim_->mtx, std::try_to_lock);
      if (lk && g_model && g_data) {
        mju_copy(qpos_buf_.data(), g_data->qpos, g_model->nq);
        mju_copy(qvel_buf_.data(), g_data->qvel, g_model->nv);
        mju_copy(qfrc_buf_.data(), g_data->qfrc_applied, g_model->nv);
        if (g_model->nsensordata > 0)
          mju_copy(sens_buf_.data(), g_data->sensordata, g_model->nsensordata);
        jd_msg_.sim_time = toTimeMsg(g_data->time);
        sd_msg_.sim_time = jd_msg_.sim_time;
      }
    }

    // 메시지 채우기 & 퍼블리시
    auto now = this->get_clock()->now();
    jd_msg_.header.stamp = now;
    sd_msg_.header.stamp = now;

    for (size_t i = 0; i < joint_slices_.size(); ++i) {
      const auto& sl = joint_slices_[i];
      std::memcpy(jd_msg_.positions[i].value.data.data(), &qpos_buf_[sl.idx_q], sizeof(double)*sl.nq);
      std::memcpy(jd_msg_.velocities[i].value.data.data(), &qvel_buf_[sl.idx_v], sizeof(double)*sl.nv);
      std::memcpy(jd_msg_.torques[i].value.data.data(),    &qfrc_buf_[sl.idx_v], sizeof(double)*sl.nv);
    }
    for (size_t i = 0; i < sensor_slices_.size(); ++i) {
      const auto& ss = sensor_slices_[i];
      if (ss.dim)
        std::memcpy(sd_msg_.sensors[i].value.data.data(), &sens_buf_[ss.idx], sizeof(double)*ss.dim);
    }
    pub_joint_dict_->publish(jd_msg_);
    pub_sensor_dict_->publish(sd_msg_);

    std::this_thread::sleep_until(next);
    while (clock::now() > next + period) next += period;
  }
}

// ---------- start/stop UI ----------
void MujocoSimNode::startSimUI(const std::string& xml_path) {
  std::unique_ptr<mj::PlatformUIAdapter> adapter;
  bool passive = false;

  if (enable_viewer_) {
    adapter = std::make_unique<mj::GlfwAdapter>();
    passive = false;
  } else {
    adapter = nullptr;
    passive = true;
  }

  sim_ = std::make_unique<mj::Simulate>(std::move(adapter), &cam_, &opt_, &pert_, passive);
  physics_thread_ = std::thread(&::PhysicsThread, sim_.get(), xml_path.c_str());
  g_sim_for_shutdown = sim_.get();
}

void MujocoSimNode::stopSimUI() {
  g_sim_for_shutdown = nullptr;
  if (sim_) sim_->exitrequest.store(true);
  if (physics_thread_.joinable()) physics_thread_.join();
  freeOffscreenContext();
  sim_.reset();
}

// ---------- model path ----------
std::string MujocoSimNode::resolveModelPath() const {
  if (!model_xml_.empty()) return model_xml_;
  if (!robot_name_.empty()) {
    try {
      const auto share = ament_index_cpp::get_package_share_directory("mujoco_ros_sim");
      return share + "/mujoco_menagerie/" + robot_name_ + "/scene.xml";
    } catch (...) {}
  }
  return "";
}

// ---------- dictionaries ----------
void MujocoSimNode::buildDictionaries() {
  joint_names_.clear(); joint_slices_.clear();
  sensor_slices_.clear();

  const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
  if (!model_) return;

  for (int i = 0; i < model_->njnt; ++i) {
    const char* nm = mj_id2name(model_, mjOBJ_JOINT, i);
    if (!nm || !*nm) continue;
    std::string jn(nm);
    joint_names_.push_back(jn);

    const int idx_q  = model_->jnt_qposadr[i];
    const int idx_v  = model_->jnt_dofadr[i];
    const int next_q = (i+1<model_->njnt)? model_->jnt_qposadr[i+1] : model_->nq;
    const int next_v = (i+1<model_->njnt)? model_->jnt_dofadr[i+1] : model_->nv;

    joint_slices_.push_back({idx_q, next_q-idx_q, idx_v, next_v-idx_v, jn});
  }

  for (int i = 0; i < model_->nsensor; ++i) {
    const int adr = model_->sensor_adr[i];
    const int dim = model_->sensor_dim[i];
    const char* nm = mj_id2name(model_, mjOBJ_SENSOR, i);
    std::string sn = nm ? std::string(nm) : ("sens"+std::to_string(i));
    sensor_slices_.push_back({adr, dim, sn});
  }
}

// ---------- image capture (UI thread only) ----------
void MujocoSimNode::buildImageSlices() {
  image_slices_.clear();

  const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
  if (!model_) return;

  int def_w = model_->vis.global.offwidth  > 0 ? model_->vis.global.offwidth  : 640;
  int def_h = model_->vis.global.offheight > 0 ? model_->vis.global.offheight : 480;

  for (int i = 0; i < model_->ncam; ++i) {
    const char* nm = mj_id2name(model_, mjOBJ_CAMERA, i);
    std::string name = (nm && *nm) ? std::string(nm) : ("camera" + std::to_string(i));
    image_slices_.push_back({ i, def_w, def_h, name });
  }
}

void MujocoSimNode::ensureOffscreenContext(int W, int H) {
  if (off_ready_ && W <= off_w_ && H <= off_h_) return;

  if (off_ready_) {
    mjv_freeScene(&off_scn_);
    mjr_freeContext(&off_con_);
    off_ready_ = false;
  }

  mjv_defaultScene(&off_scn_);
  mjr_defaultContext(&off_con_);

  mjv_makeScene(model_, &off_scn_, /*maxgeom*/ 2000);
  mjr_makeContext(model_, &off_con_, mjFONTSCALE_150);

  mjr_setBuffer(mjFB_OFFSCREEN, &off_con_);
  mjr_resizeOffscreen(std::max(1, W), std::max(1, H), &off_con_);

  off_w_ = W; off_h_ = H;
  rgb_buffer_.resize(static_cast<size_t>(off_w_) * off_h_ * 3);
  off_ready_ = true;
}

void MujocoSimNode::freeOffscreenContext() {
  if (off_ready_) {
    mjv_freeScene(&off_scn_);
    mjr_freeContext(&off_con_);
    off_ready_ = false;
    off_w_ = off_h_ = 0;
    rgb_buffer_.clear();
  }
}

void MujocoSimNode::captureCamerasOnMainIfDue() {
  if (!enable_viewer_ || !sim_) return;

  if (next_cap_.time_since_epoch().count() != 0 &&
      std::chrono::steady_clock::now() < next_cap_) {
    return;
  }
  const double period_ms = 1000.0 / camera_fps_;
  next_cap_ = std::chrono::steady_clock::now() + std::chrono::milliseconds((int)std::round(period_ms));
  if (image_slices_.empty()) return;

  int maxW = 0, maxH = 0;
  for (const auto& s : image_slices_) { maxW = std::max(maxW, s.width); maxH = std::max(maxH, s.height); }
  if (maxW <= 0 || maxH <= 0) { maxW = 640; maxH = 480; }
  ensureOffscreenContext(maxW, maxH);

  // ---------- [락: 짧게] 상태 스냅샷 ----------
  mjModel* m = nullptr;
  {
    const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
    if (!g_model || !g_data) return;

    if (d_render_model_ != g_model) {
      if (d_render_) mj_deleteData(d_render_);
      d_render_model_ = g_model;
      if (d_render_model_) d_render_ = mj_makeData(d_render_model_);
    }

    m = d_render_model_;

    mju_copy(d_render_->qpos, g_data->qpos, m->nq);
    mju_copy(d_render_->qvel, g_data->qvel, m->nv);
    if (m->na > 0) mju_copy(d_render_->act, g_data->act, m->na);
    d_render_->time = g_data->time;
  } // ---------- [락 해제] ----------

  mj_forward(m, d_render_);

  mujoco_ros_sim_msgs::msg::ImageDict idict;
  idict.header.stamp = this->get_clock()->now();
  idict.sim_time     = toTimeMsg(d_render_->time);

  mjvOption opt = opt_;
  mjvPerturb per = pert_;

  mjr_setBuffer(mjFB_OFFSCREEN, &off_con_);
  idict.images.reserve(image_slices_.size());

  for (const auto& s : image_slices_) {
    mjvCamera cam = cam_;
    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = s.cam_id;

    mjv_updateScene(m, d_render_, &opt, &per, &cam, mjCAT_ALL, &off_scn_);

    mjrRect vp{0, 0, s.width, s.height};
    if (s.width > off_con_.offWidth || s.height > off_con_.offHeight) {
      mjr_resizeOffscreen(s.width, s.height, &off_con_);
    }

    // std::vector<float> depth_buf;
    // depth_buf.resize(static_cast<size_t>(s.width) * s.height);

    mjr_render(vp, &off_scn_, &off_con_);

    const size_t nbytes_rgb = static_cast<size_t>(s.width) * s.height * 3;
    if (rgb_buffer_.size() < nbytes_rgb) rgb_buffer_.resize(nbytes_rgb);

    std::vector<float> depth_buf(static_cast<size_t>(s.width) * s.height);
    mjr_readPixels(rgb_buffer_.data(), depth_buf.data(), vp, &off_con_);

    // 카메라 frustum
    const float znear = off_scn_.camera ? off_scn_.camera->frustum_near : 0.01f;
    const float zfar  = off_scn_.camera ? off_scn_.camera->frustum_far  : 50.0f;

    // depth → meters
    std::vector<float> depth_m(depth_buf.size());
    const float eps = 1e-6f;
    for (size_t i = 0; i < depth_m.size(); ++i) {
      float z = std::min(1.0f - eps, std::max(eps, depth_buf[i]));
      const float z_ndc = 2.0f*z - 1.0f;
      const float denom = (zfar + znear) - z_ndc*(zfar - znear);
      depth_m[i] = (2.0f * znear * zfar) / denom;
    }
    sensor_msgs::msg::Image rgb_image;
    rgb_image.header.stamp    = idict.header.stamp;
    rgb_image.header.frame_id = s.name;
    rgb_image.height = s.height;
    rgb_image.width  = s.width;
    rgb_image.encoding = "rgb8";
    rgb_image.is_bigendian = false;
    rgb_image.step = s.width * 3;
    rgb_image.data.assign(rgb_buffer_.begin(), rgb_buffer_.begin() + static_cast<std::ptrdiff_t>(nbytes_rgb));

    sensor_msgs::msg::Image depth_img;
    depth_img.header.stamp = idict.header.stamp;
    depth_img.header.frame_id = s.name;           // 프레임은 RGB와 동일
    depth_img.height = s.height;
    depth_img.width  = s.width;
    depth_img.encoding = "32FC1";
    depth_img.is_bigendian = false;
    depth_img.step = s.width * sizeof(float);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(depth_m.data());
    depth_img.data.assign(p, p + depth_m.size()*sizeof(float));

    mujoco_ros_sim_msgs::msg::NamedImage nim;
    nim.name  = s.name;
    nim.rgb_image = std::move(rgb_image);
    nim.depth_image = std::move(depth_img);
    idict.images.push_back(std::move(nim));
  }

  pub_image_dict_->publish(std::move(idict));

  auto& win_con = sim_->platform_ui->mjr_context();
  mjr_setBuffer(mjFB_WINDOW, &win_con);
}

void MujocoSimNode::prepareMsgsOnce() {
  jd_msg_.positions.clear();
  jd_msg_.velocities.clear();
  jd_msg_.torques.clear();
  jd_msg_.positions.reserve(joint_slices_.size());
  jd_msg_.velocities.reserve(joint_slices_.size());
  jd_msg_.torques.reserve(joint_slices_.size());

  for (const auto& sl : joint_slices_) {
    mujoco_ros_sim_msgs::msg::NamedFloat64Array pos, vel, tau;
    pos.name = sl.name; pos.value.data.resize(sl.nq);
    vel.name = sl.name; vel.value.data.resize(sl.nv);
    tau.name = sl.name; tau.value.data.resize(sl.nv);
    jd_msg_.positions.push_back(std::move(pos));
    jd_msg_.velocities.push_back(std::move(vel));
    jd_msg_.torques.push_back(std::move(tau));
  }

  sd_msg_.sensors.clear();
  sd_msg_.sensors.reserve(sensor_slices_.size());
  for (const auto& ss : sensor_slices_) {
    mujoco_ros_sim_msgs::msg::NamedFloat64Array sm;
    sm.name = ss.name; sm.value.data.resize(ss.dim);
    sd_msg_.sensors.push_back(std::move(sm));
  }

  qpos_buf_.resize(model_->nq);
  qvel_buf_.resize(model_->nv);
  qfrc_buf_.resize(model_->nv);
  sens_buf_.resize(model_->nsensordata);
}

// ---------- ctrl / joint_state ----------
void MujocoSimNode::onCtrlMsg(const mujoco_ros_sim_msgs::msg::CtrlDict::SharedPtr msg) {
  if (!sim_) return;
  const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
  if (!g_model || !g_data) return;

  for (const auto& cmd : msg->commands) {
    int aid = -1;
    for (int i=0; i<g_model->nu; ++i) {
      const char* nm = mj_id2name(g_model, mjOBJ_ACTUATOR, i);
      if (nm && cmd.name == nm) { aid = i; break; }
    }
    if (aid < 0 || aid >= g_model->nu) continue;

    double v = 0.0;
    if (!cmd.value.data.empty()) v = cmd.value.data.front();
    g_data->ctrl[aid] = v;
  }
}

void MujocoSimNode::publishJointState() {
  if (joint_slices_.empty()) return;

  sensor_msgs::msg::JointState js;
  js.header.stamp = this->get_clock()->now();
  js.name = joint_names_;

  const std::unique_lock<std::recursive_mutex> lk(sim_->mtx);
  if (!g_model || !g_data) return;

  for (const auto& sl : joint_slices_) {
    for (int k=0; k<sl.nq; ++k) js.position.push_back(g_data->qpos[sl.idx_q + k]);
    for (int k=0; k<sl.nv; ++k) js.velocity.push_back(g_data->qvel[sl.idx_v + k]);
  }

  pub_joint_state_->publish(js);
}

void MujocoSimNode::runUiBlocking() {
  if (enable_viewer_ && sim_) {
    sim_->RenderLoop();
  }
}

} // namespace mujoco_ros_sim


// ========================== 샘플 보일러플레이트 구현 ==========================
namespace {

const int kErrorLength = 1024;
using Seconds = std::chrono::duration<double>;

void scanPluginLibraries() {
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i=0;i<nplugin;++i)
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
  }

  const char sep = '/';
  std::string exe_dir;

  {
    char buf[1024]; ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (n>0) { buf[n]='\0'; exe_dir = buf; }
    auto p = exe_dir.find_last_of(sep);
    if (p!=std::string::npos) exe_dir = exe_dir.substr(0,p);
  }

  if (exe_dir.empty()) return;

  const std::string plugin_dir = exe_dir + sep + "mujoco_plugin";
  mj_loadAllPluginLibraries(plugin_dir.c_str(),
    +[](const char* filename, int first, int count){
      std::printf("Plugins registered by library '%s':\n", filename);
      for (int i=first; i<first+count; ++i)
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    });
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  if (!filename[0]) return nullptr;

  char loadError[kErrorLength] = "";
  mjModel* mnew = nullptr;
  auto load_start = mj::Simulate::Clock::now();

  std::string ext;
  {
    std::string name(filename);
    auto dot = name.rfind('.');
    if (dot != std::string::npos && dot+1 < name.size()) ext = name.substr(dot);
  }

  if (ext == ".mjb") {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) mju::strcpy_arr(loadError, "could not load binary model");
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
    if (loadError[0]) {
      int len = mju::strlen_arr(loadError);
      if (len>0 && loadError[len-1]=='\n') loadError[len-1]='\0';
    }
  }

  double load_seconds = Seconds(mj::Simulate::Clock::now()-load_start).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  } else if (!loadError[0] && load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  } else if (loadError[0]) {
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  mju::strcpy_arr(sim.load_error, loadError);
  return mnew;
}

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

void PhysicsLoop(mj::Simulate& sim) {
  using Clock = mj::Simulate::Clock;
  std::chrono::time_point<Clock> syncCPU;
  mjtNum syncSim = 0;

  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = mnew ? mj_makeData(mnew) : nullptr;
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(g_data);
        mj_deleteModel(g_model);
        g_model = mnew;
        g_data  = dnew;
        mj_forward(g_model, g_data);
      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = mnew ? mj_makeData(mnew) : nullptr;
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(g_data);
        mj_deleteModel(g_model);
        g_model = mnew;
        g_data  = dnew;
        mj_forward(g_model, g_data);
      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.run && sim.busywait) std::this_thread::yield();
    else                         std::this_thread::sleep_for(std::chrono::milliseconds(1));

    const std::unique_lock<std::recursive_mutex> lock(sim.mtx);
    if (!g_model) continue;

    if (sim.run) {
      bool stepped = false;
      const auto startCPU = Clock::now();
      const auto elapsedCPU = startCPU - syncCPU;
      double elapsedSim = g_data->time - syncSim;
      double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

      bool misaligned =
        std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > 0.1;

      if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
          syncCPU.time_since_epoch().count()==0 || misaligned || sim.speed_changed) {
        syncCPU = startCPU; syncSim = g_data->time; sim.speed_changed = false;
        mj_step(g_model, g_data);
        if (const char* msg = Diverged(g_model->opt.disableflags, g_data)) {
          sim.run = 0; mju::strcpy_arr(sim.load_error, msg);
        } else stepped = true;
      } else {
        bool measured = false;
        auto prevSim = g_data->time;
        double refreshTime = 0.7 / sim.refresh_rate;

        while (Seconds((g_data->time - syncSim)*slowdown) < Clock::now() - syncCPU &&
               Clock::now() - startCPU < Seconds(refreshTime)) {

          if (!measured && elapsedSim) {
            sim.measured_slowdown =
              std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
            measured = true;
          }

          sim.InjectNoise();
          mj_step(g_model, g_data);
          if (const char* msg = Diverged(g_model->opt.disableflags, g_data)) {
            sim.run = 0; mju::strcpy_arr(sim.load_error, msg);
          } else stepped = true;

          if (g_data->time < prevSim) break;
        }
      }

      if (stepped) sim.AddToHistory();
    } else {
      mj_forward(g_model, g_data);
      if (sim.pause_update) mju_copy(g_data->qacc_warmstart, g_data->qacc, g_model->nv);
      sim.speed_changed = true;
    }
  }
}

} // anonymous


// --------------------------- main ---------------------------
int main(int argc, char** argv) {
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) mju_error("Headers and library have different versions");

  rclcpp::init(argc, argv);
  auto node = std::make_shared<mujoco_ros_sim::MujocoSimNode>();

  rclcpp::on_shutdown([](){
    if (auto *sim = g_sim_for_shutdown.load()) {
      sim->exitrequest.store(true);
    }
  });

  // ROS 실행을 별도 스레드로
  std::thread ros_thread([&](){
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
    exec.add_node(node);
    exec.spin();
  });

  // 메인 스레드에서 UI 루프 실행
  node->runUiBlocking();

  rclcpp::shutdown();
  if (ros_thread.joinable()) ros_thread.join();
  return 0;
}
