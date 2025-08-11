#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 40

#include <boost/python.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/python/numpy.hpp>
#include <eigenpy/eigenpy.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <dlfcn.h>
#include <memory>
#include <stdexcept>

#include "mujoco_ros_sim/ControllerInterface.hpp"
#include "mujoco_ros_sim/ControllerRegistry.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;
using ControllerSP      = std::shared_ptr<ControllerInterface>;
using ControllerFactory = std::function<std::unique_ptr<ControllerInterface>()>;
using SigT              = boost::mpl::vector<ControllerSP>;


static VecMap pyDict_to_VecMap(const bp::dict &py)
{
  VecMap out;
  bp::list keys = py.keys();
  for (Py_ssize_t i = 0; i < bp::len(keys); ++i)
  {
    std::string key = bp::extract<std::string>( keys[i] );
    Eigen::VectorXd vec = bp::extract<Eigen::VectorXd>( py[keys[i]] );
    out.emplace(std::move(key), std::move(vec));
  }
  return out;
}

void updateState_wrapper(ControllerInterface &self,
                         const bp::dict &pos,
                         const bp::dict &vel,
                         const bp::dict &tau,
                         const bp::dict &sens,
                         double t)
{
  self.updateState( pyDict_to_VecMap(pos),
                    pyDict_to_VecMap(vel),
                    pyDict_to_VecMap(tau),
                    pyDict_to_VecMap(sens),
                    t );
}

static bp::dict map_to_pydict(const CtrlInputMap &m)
{
  bp::dict out;
  for (const auto &kv : m)
    out[kv.first] = kv.second;
  return out;
}

static bp::dict getCtrlInput_wrapper(const ControllerInterface &self)
{
  return map_to_pydict( self.getCtrlInput() );
}

static cv::Mat ndarray_to_mat(const np::ndarray& arr)
{
  // dtype/shape/stride
  const int nd = arr.get_nd();
  const Py_intptr_t* shp = arr.get_shape();
  const Py_intptr_t* str = arr.get_strides();

  // RGB8: (H,W,3) uint8
  if (nd == 3 &&
      arr.get_dtype() == np::dtype::get_builtin<uint8_t>() &&
      shp[2] == 3)
  {
    int h = static_cast<int>(shp[0]);
    int w = static_cast<int>(shp[1]);
    // row stride(bytes) = str[0]
    auto* data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(arr.get_data()));
    return cv::Mat(h, w, CV_8UC3, data, static_cast<size_t>(str[0]));
  }

  // MONO8: (H,W) uint8
  if (nd == 2 && arr.get_dtype() == np::dtype::get_builtin<uint8_t>())
  {
    int h = static_cast<int>(shp[0]);
    int w = static_cast<int>(shp[1]);
    auto* data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(arr.get_data()));
    return cv::Mat(h, w, CV_8UC1, data, static_cast<size_t>(str[0]));
  }

  // DEPTH32F: (H,W) float32
  if (nd == 2 && arr.get_dtype() == np::dtype::get_builtin<float>())
  {
    int h = static_cast<int>(shp[0]);
    int w = static_cast<int>(shp[1]);
    auto* data = const_cast<float*>(reinterpret_cast<const float*>(arr.get_data()));
    return cv::Mat(h, w, CV_32FC1, data, static_cast<size_t>(str[0]));
  }

  PyErr_SetString(PyExc_TypeError, "Unsupported ndarray shape/dtype for cv::Mat");
  bp::throw_error_already_set();
  return {};
}

static ImageCVMap pydict_to_imagemap_cv(const bp::dict& d)
{
  ImageCVMap out;
  bp::list keys = d.keys();
  for (Py_ssize_t i = 0; i < bp::len(keys); ++i) 
  {
    bp::object k = keys[i];
    std::string name = bp::extract<std::string>(k);
    np::ndarray arr = bp::extract<np::ndarray>(d[k]);
    out.emplace(name, ndarray_to_mat(arr));
  }
  return out;
}

static void updateRGBDImage_cv_wrapper(ControllerInterface& self, const bp::dict& rgbd)
{
  self.updateRGBDImage( pydict_to_imagemap_cv(rgbd) );
}

static void load_plugin_library(const std::string& path)
{
  if (!dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL))
      throw std::runtime_error(dlerror());
}

static void export_new_factories()
{
  bp::object mod = bp::scope();
  auto& registry = ControllerRegistry::instance().map();

  for (auto& kv : registry)
  {
    const std::string pyname = kv.first + "_cpp";
    if (PyObject_HasAttrString(mod.ptr(), pyname.c_str()))
      continue;

    ControllerFactory fac = kv.second;
    auto wrapper = [fac]()->ControllerSP
    { return ControllerSP(fac().release()); };

    bp::object pyfunc = bp::make_function(wrapper,
                                      bp::default_call_policies(),
                                      SigT());

    bp::object top = bp::import("bindings");
    PyObject_SetAttrString(top.ptr(),  pyname.c_str(), pyfunc.ptr());

    bp::object pkg = bp::import("mujoco_ros_sim.bindings");
    PyObject_SetAttrString(pkg.ptr(),  pyname.c_str(), pyfunc.ptr());
  }
}


BOOST_PYTHON_MODULE(bindings)
{
  eigenpy::enableEigenPy();
  np::initialize();

  bp::register_ptr_to_python< std::shared_ptr<ControllerInterface> >();
  bp::class_<ControllerInterface, std::shared_ptr<ControllerInterface>, boost::noncopyable >("ControllerInterface", bp::no_init)
    .def("starting",        &ControllerInterface::starting)
    .def("updateState",     &updateState_wrapper)
    .def("updateRGBDImage", &updateRGBDImage_cv_wrapper)
    .def("compute",         &ControllerInterface::compute)
    .def("getCtrlInput",    &getCtrlInput_wrapper)
    .def("getCtrlTimeStep", &ControllerInterface::getCtrlTimeStep);

  export_new_factories();

  bp::def("load_plugin_library",   &load_plugin_library,  bp::args("path"));
  bp::def("export_new_factories",  &export_new_factories);
}
