#include "mujoco_ros_sim/mujoco_controller.hpp"
#include "mujoco_ros_sim/utils.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <iomanip>
#include <sstream>

// shorter aliases
using mujoco_ros_sim_msgs::msg::JointDict;
using mujoco_ros_sim_msgs::msg::SensorDict;
using mujoco_ros_sim_msgs::msg::CtrlDict;
using mujoco_ros_sim_msgs::msg::NamedFloat64Array;
using mujoco_ros_sim_msgs::msg::ImageDict;

// ---------- helpers ----------
namespace {

// value: Float64MultiArray → Eigen::VectorXd
inline Eigen::VectorXd eigenFromNamed(const NamedFloat64Array& m)
{
    // to Eigen
    const auto& data = m.value.data;
    Eigen::VectorXd v(static_cast<Eigen::Index>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) v(static_cast<Eigen::Index>(i)) = data[i];
    return v;
}

// command scalar → NamedFloat64Array
inline NamedFloat64Array namedFromScalar(const std::string& name, double val)
{
    // one value
    NamedFloat64Array out;
    out.name = name;
    out.value.layout.dim.clear();
    out.value.layout.data_offset = 0;
    out.value.data.resize(1);
    out.value.data[0] = val;
    return out;
}

// sensor_msgs/Image → cv::Mat
inline cv::Mat cvMatFromImageMsg(const sensor_msgs::msg::Image& img)
{
    // cv bridge
    auto cv_ptr = cv_bridge::toCvCopy(img, img.encoding);
    return cv_ptr->image;
}

} // namespace

// ---------- class impl ----------

ControllerNode::ControllerNode()
: Node("controller_node")
{
    // param read
    controller_class_ = declare_parameter<std::string>("controller_class", "mujoco_ros_sim/py_controller");
}

void ControllerNode::init_controller() 
{
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.best_effort();
    qos.durability_volatile();

    // ---- unify spec ----
    // Always accept 'pkg/Class' for both C++ and Python.
    // Try pluginlib first; if not available -> fallback to PyController with python_class="pkg:Class".
    std::string spec = this->get_parameter("controller_class").as_string(); // already declared in ctor

    // Optional '@path' → python_path
    std::string python_path;
    if (auto at = spec.find('@'); at != std::string::npos) {
        python_path = spec.substr(at + 1);
        spec.erase(at);
    }

    // Decide load target
    std::string plugin_to_load = spec;       // try as C++ plugin
    std::string python_class;                // for fallback
    bool use_python = false;

    // If spec looks like 'pkg/Class' but plugin not found, fallback to Python
    try 
    {
        // Fast check if available (prefer, if you have it)
        if (!(loader_.isClassAvailable(spec))) use_python = true;
    } 
    catch (...) 
    {
        // Some loaders don’t expose isClassAvailable cleanly; ignore and try create
    }

    if (!use_python) 
    {
        try 
        {
            auto factory = loader_.createSharedInstance(plugin_to_load);
            controller_  = factory->create(shared_from_this());
        } 
        catch (...) 
        {
            // Not a C++ plugin → Python fallback
            use_python = true;
        }
    }

    if (use_python) 
    {
        // Map 'pkg/Class' -> 'pkg:Class'
        auto slash = spec.find('/');
        if (slash == std::string::npos) throw std::runtime_error("controller_class must be 'pkg/Class' (got: " + spec + ")");
        python_class = spec.substr(0, slash) + ":" + spec.substr(slash + 1);
        plugin_to_load = "mujoco_ros_sim/py_controller";

        if (!this->has_parameter("python_class")) this->declare_parameter<std::string>("python_class", "");
        if (!this->has_parameter("python_path"))  this->declare_parameter<std::string>("python_path", "");

        // Set params for PyController (auto)
        std::vector<rclcpp::Parameter> params;
        params.emplace_back("python_class", rclcpp::ParameterValue(python_class));
        if (!python_path.empty()) params.emplace_back("python_path",  rclcpp::ParameterValue(python_path));
        this->set_parameters(params);

        // Load PyController
        auto factory = loader_.createSharedInstance(plugin_to_load);
        controller_  = factory->create(shared_from_this());
    }

    RCLCPP_INFO(get_logger(), "%sController loaded: %s%s", cblue_, plugin_to_load.c_str(), creset_);

    const double dt = controller_->getCtrlTimeStep();
    period_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

    timer_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    sub_group_   = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions opt;
    opt.callback_group = sub_group_;
    joint_sub_  = create_subscription<JointDict>("mujoco_ros_sim/joint_dict", qos,
                    std::bind(&ControllerNode::jointCb, this, std::placeholders::_1), opt);
    sensor_sub_ = create_subscription<SensorDict>("mujoco_ros_sim/sensor_dict", qos,
                    std::bind(&ControllerNode::sensorCb, this, std::placeholders::_1), opt);
    image_sub_  = create_subscription<ImageDict>("mujoco_ros_sim/image_dict", qos,
                    std::bind(&ControllerNode::imageCb, this, std::placeholders::_1), opt);
    ctrl_pub_   = create_publisher<CtrlDict>("mujoco_ros_sim/ctrl_dict", qos);

    // timer_ = create_wall_timer(period_ns_, std::bind(&ControllerNode::controlLoop, this), timer_group_);
    // ROS 타이머 대신 고정주기 스레드(지터↓, 1kHz 수렴)
    rt_run_.store(true, std::memory_order_relaxed);
    rt_thread_ = std::thread(&ControllerNode::run_rt_loop, this);
}

ControllerNode::~ControllerNode() 
{
  rt_run_.store(false, std::memory_order_relaxed);
  if (rt_thread_.joinable()) rt_thread_.join();
}

void ControllerNode::run_rt_loop() 
{
    using clock = std::chrono::steady_clock;
    // (선택) 실시간 스케줄링/메모리 잠금 시도
    try_set_realtime(60);

    const auto period = period_ns_;
    auto next = clock::now();

    while (rt_run_.load(std::memory_order_relaxed) && rclcpp::ok()) 
    {
        next += period;

        // 기존 제어 1 step
        controlLoop();

        // 절대 데드라인까지 슬립 (지터 최소화)
        std::this_thread::sleep_until(next);

        // 백로그가 쌓였으면 누적 보정
        while (clock::now() > next + period) next += period;
    }
}

void ControllerNode::try_set_realtime(int prio) 
{
    // 메모리 스와핑 방지 시도 (실패해도 무시)
    mlockall(MCL_CURRENT | MCL_FUTURE);

    sched_param sp{};
    sp.sched_priority = prio;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);  // 권한 없으면 실패해도 진행
    // CPU 고정 등 더 하려면 pthread_setaffinity_np(...) 사용 가능
}



void ControllerNode::jointCb(const JointDict::SharedPtr msg)
{
    // joints
    VecMap pos, vel, tau;
    for (const auto& it : msg->positions)  pos[it.name] = eigenFromNamed(it);
    for (const auto& it : msg->velocities) vel[it.name] = eigenFromNamed(it);
    for (const auto& it : msg->torques)    tau[it.name] = eigenFromNamed(it);
    
    const double sim_time = rclcpp::Time(msg->sim_time).seconds();
    
    {
        // swap in
        std::scoped_lock lk(state_mtx_);
        latest_pos_   = std::move(pos);
        latest_vel_   = std::move(vel);
        latest_tau_   = std::move(tau);
        latest_time_  = sim_time;
        have_joint_   = true;
    }

}

void ControllerNode::sensorCb(const SensorDict::SharedPtr msg)
{
    // sensors
    VecMap sensors;

    for (const auto& it : msg->sensors) sensors[it.name] = eigenFromNamed(it);
    {
        // swap in
        std::scoped_lock lk(state_mtx_);
        latest_sensors_ = std::move(sensors);
        have_sensor_ = true;
    }
}

void ControllerNode::imageCb(const ImageDict::SharedPtr msg)
{
  try
  {
    ImageCVMap imgs;
    imgs.reserve(msg->images.size());

    for (const auto& ni : msg->images)
    {
      RGBDFrame f;

      // RGB
      f.rgb = cvMatFromImageMsg(ni.rgb_image);  // CV_8UC3 (rgb8/bgr8 등)

      // Depth
      if (!ni.depth_image.data.empty()) 
      {
        // 32FC1 or 16UC1 -> cv::Mat
        f.depth = cvMatFromImageMsg(ni.depth_image);
      } 
      else 
      {
        f.depth.release();
      }

      imgs.emplace(ni.name, std::move(f));
    }

    controller_->updateRGBDImage(imgs);
    (void)msg->sim_time;
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(get_logger(), "imageCb error: %s", e.what());
  }
}


void ControllerNode::controlLoop()
{
    // t0
    const auto t0 = std::chrono::steady_clock::now();

    // snapshot
    VecMap pos, vel, tau, sensors;
    double sim_time = 0.0;
    {
        std::scoped_lock lk(state_mtx_);

        if (!(have_joint_)) {return;} // wait
        pos = latest_pos_; 
        vel = latest_vel_; 
        tau = latest_tau_;
        sensors = latest_sensors_; 
        sim_time = latest_time_;
    }
    // t1
    const auto t1 = std::chrono::steady_clock::now();

    // push state
    controller_->updateState(pos, vel, tau, sensors, sim_time);

    // first tick
    if (!started_) 
    { 
        controller_->starting(); 
        started_ = true; 
    }
    // t2
    const auto t2 = std::chrono::steady_clock::now();

    // run compute
    controller_->compute();
    // t3
    const auto t3 = std::chrono::steady_clock::now();

    // publish
    auto cmd_map = controller_->getCtrlInput();
    CtrlDict out;
    out.header.stamp = this->get_clock()->now();
    out.commands.reserve(cmd_map.size());
    for (const auto& kv : cmd_map) out.commands.emplace_back(namedFromScalar(kv.first, kv.second));
    ctrl_pub_->publish(out);
    // t4
    const auto t4 = std::chrono::steady_clock::now();

    // timing
    const double getData_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double updateState_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double compute_ms     = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double applyCtrl_ms   = std::chrono::duration<double, std::milli>(t4 - t3).count();
    const double total_ms       = std::chrono::duration<double, std::milli>(t4 - t0).count();

    // overrun check
    const double dt_s   = std::chrono::duration<double>(period_ns_).count();
    const double step_s = total_ms / 1000.0;
    if (dt_s - step_s < 0.0) 
    {
        std::ostringstream oss;
        oss << "\n===================================\n"
        << "getData took "     << std::fixed << std::setprecision(6) << getData_ms     << " ms\n"
        << "updateState took " << std::fixed << std::setprecision(6) << updateState_ms << " ms\n"
        << "compute took "     << std::fixed << std::setprecision(6) << compute_ms     << " ms\n"
        << "applyCtrl took "   << std::fixed << std::setprecision(6) << applyCtrl_ms   << " ms\n"
        << "totalStep took "   << std::fixed << std::setprecision(6) << total_ms       << " ms\n"
        << "===================================";
        RCLCPP_WARN(get_logger(), "%s", oss.str().c_str());
    }
}

int main(int argc, char** argv) 
{
    // init
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControllerNode>();
    node->init_controller();

    // executor
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
    exec.add_node(node);
    exec.spin();

    // shutdown
    rclcpp::shutdown();
    return 0;
}
