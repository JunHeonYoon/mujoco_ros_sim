#include "mujoco_ros_sim/PyController.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <dlfcn.h>
#include <Python.h>
#include <atomic>
#include <memory>

namespace py = pybind11;


// promote libpython
static void promote_libpython_to_global() 
{
  static bool done = false;
  if (done) return;
  done = true;

  const char* cands[] = {
    "libpython3.10.so.1.0",
    "libpython3.10.so",
    nullptr
  };
  for (const char** p = cands; *p; ++p) 
  {
    if (void* h = dlopen(*p, RTLD_NOW | RTLD_GLOBAL)) 
    {
      // global symbols
      return;
    }
  }
  // optional warn
}

namespace 
{
  // init Python
  void ensure_python() 
  {
    static bool inited = false;
    if (!inited) 
    {
      ::setenv("PYTHONNOUSERSITE", "1", 1);
      py::initialize_interpreter();

      // release GIL
      PyEval_SaveThread();

      inited = true;
    }
  }

  // Eigen → numpy
  py::array eigen_to_np(const Eigen::VectorXd& v) 
  {
    auto a = py::array_t<double>(v.size());
    std::memcpy(a.mutable_data(), v.data(), sizeof(double) * v.size());
    return a;
  }

  // map → dict
  py::dict vecmap_to_pydict(const VecMap& m) 
  {
    py::dict d;
    for (const auto& kv : m) d[py::str(kv.first)] = eigen_to_np(kv.second);
    return d;
  }

  // cv helper
  template<typename T>
  py::array cv_to_np_impl(const cv::Mat& img) 
  {
    const int h = img.rows, w = img.cols, c = img.channels();
    auto out = (c == 1) ? py::array_t<T>({h, w})
                        : py::array_t<T>({h, w, c});
    if (img.isContinuous()) 
    {
      std::memcpy(out.mutable_data(), img.data, img.total() * img.elemSize());
    } 
    else 
    {
      cv::Mat tmp = img.clone();
      std::memcpy(out.mutable_data(), tmp.data, tmp.total() * tmp.elemSize());
    }
    return out;
  }

  // cv → numpy
  py::object cv_to_np(const cv::Mat& img) 
  {
    switch (img.depth()) 
    {
      case CV_8U:  return cv_to_np_impl<uint8_t>(img);
      case CV_16U: return cv_to_np_impl<uint16_t>(img);
      case CV_32F: return cv_to_np_impl<float>(img);
      case CV_64F: return cv_to_np_impl<double>(img);
      default:     return cv_to_np_impl<uint8_t>(img);
    }
  }

  // image map → dict
  py::dict imagemap_to_pydict(const ImageCVMap& m) 
  {
    py::dict d;
    for (const auto& kv : m) d[py::str(kv.first)] = cv_to_np(kv.second);
    return d;
  }

}

// numpy view
static py::object make_numpy_view(std::vector<double>& buf) 
{
  return py::array(py::dtype::of<double>(),
                   { (py::ssize_t)buf.size() },
                   { (py::ssize_t)sizeof(double) },
                   buf.data(),
                   /* base = */ py::none());
}

// runtime tune
static void set_low_jitter_runtime_once() 
{
  static std::once_flag once;
  std::call_once(once, []{
    ::setenv("OMP_NUM_THREADS",       "1", 1);
    ::setenv("OPENBLAS_NUM_THREADS",  "1", 1);
    ::setenv("MKL_NUM_THREADS",       "1", 1);
    ::setenv("NUMEXPR_NUM_THREADS",   "1", 1);

    py::gil_scoped_acquire gil;
    py::module_::import("gc").attr("disable")();
  });
}

// impl data
struct PyController::Impl 
{
  // python handles
  py::object rclpy_mod, py_node, py_ctrl;
  // bound callables
  py::object fn_spin_once, fn_update_state, fn_compute, fn_get_ctrl, fn_update_img;

  // state buffers
  struct Buf { std::vector<double> v; py::object np; };
  std::unordered_map<std::string, Buf> pos, vel, tau, sen;
  py::dict py_pos, py_vel, py_tau, py_sen;
  double latest_time = 0.0;
  bool   layout_alloc = false;
  bool   py_bound     = false;
  std::mutex state_mtx;

  // image buffers
  std::mutex img_mtx;
  ImageCVMap pending_images;
  bool have_images = false;

  // command result
  std::shared_ptr<const CtrlInputMap> cmd_ptr;

  // worker thread
  std::atomic_bool run_worker{true};
  std::thread worker;
  std::atomic_bool need_start{false};
};


// ctor
PyController::PyController(const rclcpp::Node::SharedPtr& node)
: ControllerInterface(node)
{
  std::string py_class;
  if (node_->has_parameter("python_class")) {
    py_class = node_->get_parameter("python_class").as_string();
  } else {
    py_class = node_->declare_parameter<std::string>("python_class", "");
  }

  std::string py_path;
  if (node_->has_parameter("python_path")) {
    py_path = node_->get_parameter("python_path").as_string();
  } else {
    py_path = node_->declare_parameter<std::string>("python_path", "");
  }
  if (py_class.empty()) throw std::runtime_error("PyController: parameter 'python_class' must be set (e.g. 'my_controller_py.my_controller:MyController')");
  promote_libpython_to_global();
  ensure_python();
  try 
  {
    py::gil_scoped_acquire gil;
    set_low_jitter_runtime_once();
    impl_ = std::make_unique<Impl>();

    // rclpy init
    impl_->rclpy_mod = py::module_::import("rclpy");
    impl_->rclpy_mod.attr("init")();

    if (!py_path.empty()) 
    {
      py::module_ sys = py::module_::import("sys");
      sys.attr("path").cast<py::list>().append(py_path);
    }

    // parse class
    std::string mod, cls;
    {
      auto pos = py_class.find(':');
      if (pos != std::string::npos) 
      { 
        mod = py_class.substr(0, pos); 
        cls = py_class.substr(pos+1); 
      }
      else 
      {
        auto d = py_class.rfind('.');
        if (d == std::string::npos) throw std::runtime_error("python_class must be 'module:Class' or 'module.Class'");
        mod = py_class.substr(0, d); cls = py_class.substr(d+1);
      }
    }

    // make node
    const std::string parent_name = node_->get_name();
    const std::string parent_ns   = node_->get_namespace();
    const std::string py_name     = "py_" + parent_name;
    impl_->py_node = impl_->rclpy_mod.attr("create_node")(
        py::str(py_name),
        py::arg("namespace") = py::str(parent_ns.empty() ? "/" : parent_ns)
    );

    // import ctrl
    py::object mod_obj = py::module_::import(mod.c_str());
    py::object cls_obj = mod_obj.attr(cls.c_str());
    impl_->py_ctrl = cls_obj(impl_->py_node);
    impl_->fn_spin_once    = impl_->rclpy_mod.attr("spin_once");
    impl_->fn_update_state = impl_->py_ctrl.attr("updateState");
    impl_->fn_compute      = impl_->py_ctrl.attr("compute");
    impl_->fn_get_ctrl     = impl_->py_ctrl.attr("getCtrlInput");
    impl_->fn_update_img   = impl_->py_ctrl.attr("updateRGBDImage");

    // sync dt
    try { this->dt_ = impl_->py_ctrl.attr("getCtrlTimeStep")().cast<double>(); } catch (...) {}

    // worker loop
    impl_->run_worker = true;
    impl_->worker = std::thread([this](){
      set_low_jitter_runtime_once();

      const double dt = this->dt_ > 0.0 ? this->dt_ : 0.001;   // rate
      auto next   = std::chrono::steady_clock::now();
      const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(dt));

      while (impl_->run_worker.load(std::memory_order_relaxed)) 
      {
        next += period;
        try 
        {
          py::gil_scoped_acquire gil;
          if (impl_->need_start.exchange(false)) impl_->py_ctrl.attr("starting")();
          impl_->fn_spin_once(impl_->py_node, py::arg("timeout_sec") = 0.0);
          {
            std::scoped_lock lk(impl_->img_mtx);
            if (impl_->have_images) 
            {
              impl_->fn_update_img(imagemap_to_pydict(impl_->pending_images));
              impl_->have_images = false;
            }
          }
          {
            std::scoped_lock lk(impl_->state_mtx);
            if (impl_->layout_alloc) 
            {
              if (!impl_->py_bound) 
              {
                auto bind = [](auto& src, py::dict& py_out)
                {
                  for (auto& kv : src) 
                  {
                    kv.second.np = make_numpy_view(kv.second.v);   // zero-copy
                    py_out[py::str(kv.first)] = kv.second.np;
                  }
                };
                py::module_::import("numpy");
                bind(impl_->pos, impl_->py_pos);
                bind(impl_->vel, impl_->py_vel);
                bind(impl_->tau, impl_->py_tau);
                bind(impl_->sen, impl_->py_sen);
                impl_->py_bound = true;
              }

              impl_->fn_update_state(impl_->py_pos, impl_->py_vel,
                                    impl_->py_tau, impl_->py_sen,
                                    impl_->latest_time);
              impl_->fn_compute();

              py::dict d = impl_->fn_get_ctrl();
              auto out = std::make_shared<CtrlInputMap>();
              out->reserve(d.size());
              for (auto item : d) (*out)[item.first.cast<std::string>()] = item.second.cast<double>();
              std::shared_ptr<const CtrlInputMap> out_const = out;
              std::atomic_store_explicit(&impl_->cmd_ptr, out_const, std::memory_order_release);
            }
          }

        } 
        catch (const py::error_already_set& e) 
        {
          RCLCPP_ERROR(rclcpp::get_logger("PyController"), "[worker] %s", e.what());
        }
        std::this_thread::sleep_until(next);
      }
    });
    RCLCPP_INFO(node_->get_logger(), "[PyController] loaded python class: %s", py_class.c_str());
  } 
  catch (const py::error_already_set& e) 
  {
    // traceback log
    RCLCPP_FATAL(node_->get_logger(), "[PyController:init] Python error:\n%s", e.what());
    throw;
  }
}

// dtor
PyController::~PyController() 
{
  if (!impl_) return;
  impl_->run_worker.store(false);
  if (impl_->worker.joinable()) impl_->worker.join();
  try 
  {
    py::gil_scoped_acquire gil;
    if (impl_->rclpy_mod) impl_->rclpy_mod.attr("shutdown")();
  } catch (...) {}
}

// starting flag
void PyController::starting() 
{
  impl_->need_start.store(true, std::memory_order_relaxed);
}

// state update
void PyController::updateState(const VecMap& pos,
                               const VecMap& vel,
                               const VecMap& tau_ext,
                               const VecMap& sensors,
                               double sim_time)
{
  std::scoped_lock lk(impl_->state_mtx);

  auto alloc = [](const VecMap& src, auto& dst)
  {
    for (const auto& kv : src) 
    {
      auto& b = dst[kv.first];
      if (b.v.size() != static_cast<size_t>(kv.second.size())) 
      {
        b.v.resize(static_cast<size_t>(kv.second.size()), 0.0);
      }
    }
  };
  if (!impl_->layout_alloc)
  {
    alloc(pos, impl_->pos);
    alloc(vel, impl_->vel);
    alloc(tau_ext, impl_->tau);
    alloc(sensors, impl_->sen);
    impl_->layout_alloc = true;
  }

  auto copy = [](const VecMap& src, auto& dst)
  {
    for (const auto& kv : src) 
    {
      auto it = dst.find(kv.first);
      if (it == dst.end()) continue;
      const Eigen::VectorXd& v = kv.second;
      std::memcpy(it->second.v.data(), v.data(), sizeof(double)*v.size());
    }
  };
  copy(pos, impl_->pos);
  copy(vel, impl_->vel);
  copy(tau_ext, impl_->tau);
  copy(sensors, impl_->sen);

  impl_->latest_time = sim_time;
}


// image update
void PyController::updateRGBDImage(const ImageCVMap& images)
{
  std::scoped_lock lk(impl_->img_mtx);
  impl_->pending_images = images;   // optional move
  impl_->have_images = true;
}


// compute noop
void PyController::compute()
{
  auto p = std::atomic_load_explicit(&impl_->cmd_ptr, std::memory_order_acquire);
  if (!p) return;
  // optional checks
}

// get command
CtrlInputMap PyController::getCtrlInput() const {
  auto p = std::atomic_load_explicit(&impl_->cmd_ptr, std::memory_order_acquire);
  if (p) return *p;
  return {};
}

// plugin export
PLUGINLIB_EXPORT_CLASS(PyControllerFactory, ControllerFactory)