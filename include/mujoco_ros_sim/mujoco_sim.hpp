#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <mujoco_ros_sim_msgs/msg/joint_dict.hpp>
#include <mujoco_ros_sim_msgs/msg/sensor_dict.hpp>
#include <mujoco_ros_sim_msgs/msg/ctrl_dict.hpp>
#include <mujoco_ros_sim_msgs/msg/image_dict.hpp>

// MuJoCo sample UI
#include "mujoco/simulate.h"
#include "mujoco/glfw_adapter.h"
#include "mujoco/array_safety.h"

#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <string>
#include <GLFW/glfw3.h>

namespace mujoco_ros_sim {
  
  // 단순 조인트/센서 슬라이스
  struct JointSlice { int idx_q{}, nq{}; int idx_v{}, nv{}; std::string name; };
  struct SensorSlice { int idx{}, dim{}; std::string name; };
  struct ImageSlice { int cam_id{}, width{640}, height{480}; std::string name; };
  
class MujocoSimNode : public rclcpp::Node {
public:
    MujocoSimNode();
    ~MujocoSimNode() override;
    void runUiBlocking();

private:
  // ---- ROS pub/sub ----
  rclcpp::Publisher<mujoco_ros_sim_msgs::msg::JointDict>::SharedPtr  pub_joint_dict_;
  rclcpp::Publisher<mujoco_ros_sim_msgs::msg::SensorDict>::SharedPtr pub_sensor_dict_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr         pub_joint_state_;
  rclcpp::Publisher<mujoco_ros_sim_msgs::msg::ImageDict>::SharedPtr  pub_image_dict_;
  rclcpp::Subscription<mujoco_ros_sim_msgs::msg::CtrlDict>::SharedPtr sub_ctrl_;
   
  rclcpp::TimerBase::SharedPtr timer_pub_;
  rclcpp::TimerBase::SharedPtr timer_cam_;
  rclcpp::TimerBase::SharedPtr timer_joint_state_;
  std::shared_ptr<rclcpp::CallbackGroup> cb_group_;

  // ---- MuJoCo sample UI ----
  std::unique_ptr<mujoco::Simulate> sim_;      // UI/렌더 객체
  std::thread physics_thread_;                 // 물리 스레드
  std::thread ui_thread_;                      // RenderLoop 스레드
  std::atomic_bool ui_running_{false};

  // 샘플과 동일한 카메라/옵션/페터브
  mjvCamera cam_{};
  mjvOption opt_{};
  mjvPerturb pert_{};

  // ---- 모델/데이터 (샘플 루프가 소유/교체) ----
  // 주의: 실제 스텝/로드는 physics_thread가 수행. 우리는 읽기만 한다.
  // 동기화는 sim_->mtx 사용(샘플 simulate.h가 제공하는 재귀 mutex).
  mjModel* model_{nullptr};
  mjData*  data_{nullptr};

  // ---- 파라미터/상태 ----
  std::string robot_name_;
  std::string model_xml_;
  double dt_{0.002};           // 나중에 model_->opt.timestep로 갱신
  bool enable_viewer_{true};   // RenderLoop on/off
  double pub_rate_hz_{1.0/0.002}; // 퍼블리시 주기(모델 로딩 후 dt로 맞춤)

  // ---- 사전(조인트/센서) ----
  std::vector<std::string> joint_names_;
  std::unordered_map<std::string, int> jname_to_jid_;
  std::vector<JointSlice> joint_slices_;

  std::vector<std::string> sensor_names_;
  std::unordered_map<std::string, int> sname_to_sid_;
  std::unordered_map<std::string, int> sname_to_dim_;
  std::vector<SensorSlice> sensor_slices_;

  mujoco_ros_sim_msgs::msg::JointDict  jd_msg_;
  mujoco_ros_sim_msgs::msg::SensorDict sd_msg_;
  std::vector<double> qpos_buf_, qvel_buf_, qfrc_buf_, sens_buf_;

  std::thread pub_thread_;
  std::atomic_bool pub_run_{false};

  // ---- 내부 유틸 ----
  void startSimUI(const std::string& xml_path);
  void stopSimUI();

  void prepareMsgsOnce();
  void publishLoop1k();

  void buildDictionaries();        // model_ 기반으로 조인트/센서 슬라이스 구성
  void onCtrlMsg(const mujoco_ros_sim_msgs::msg::CtrlDict::SharedPtr msg);
  void publishBundles();           // JointDict / SensorDict
  void publishJointState();
  

  // 파일 찾기(패키지 share/menagerie/...) or explicit model_xml_
  std::string resolveModelPath() const;

  // helpers
  static builtin_interfaces::msg::Time toTimeMsg(double tsec);

  double camera_fps_{30.0};

  // 카메라/퍼블리셔/버퍼
  std::vector<ImageSlice> image_slices_;
  std::unordered_map<std::string,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> cam_pubs_;

  // 오프스크린 렌더 리소스(메인 스레드에서만 생성/사용)
  mjvScene   off_scn_{};          // 별도 씬
  mjrContext off_con_{};          // 별도 컨텍스트(오프스크린 버퍼 포함)
  bool       off_ready_{false};
  int        off_w_{0}, off_h_{0};
  std::vector<unsigned char> rgb_buffer_;

  // 캡처 주기 제어
  std::chrono::steady_clock::time_point next_cap_{};
  mjData* d_render_{nullptr};     // 렌더 전용 mjData (락 밖에서 사용)
  mjModel* d_render_model_{nullptr}; // d_render_가 어떤 모델 기준인지 추적

  // 이미지 퍼블리시용 내부 함수들
  void buildImageSlices();                 // 모델의 고정 카메라 목록/해상도 수집
  void ensureOffscreenContext(int W, int H);
  void freeOffscreenContext();
  void captureCamerasOnMainIfDue();        // ★ 스왑 직전(UI 스레드)에서 호출
};

} // namespace mujoco_ros_sim

