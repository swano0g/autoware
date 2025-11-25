#include <chrono>
#include <memory>
#include <future>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include <tier4_debug_msgs/msg/float32_stamped.hpp>
#include <tier4_debug_msgs/msg/int32_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

using namespace std::chrono_literals;


// FALLBACK NORMAL -> BACKUP
#define FALLBACK_NDT_ITER           50
#define FALLBACK_NDT_MISS_CNT       3
#define FALLBACK_NDT_SCORE_CNT      1

// ROLLBACK BACKUP -> NORMAL
#define ROLLBACK_NDT_SCORE_THRESH   2.3
#define ROLLBACK_NDT_SCORE_CNT      50

#define NDT_RESET_CNT               10


#define GNSS_VALID  0.5

class SupervisorNode : public rclcpp::Node
{
public:
  SupervisorNode()
  : Node("phyerr_supervisor_node")
  {
    // --- Parameters ---
    ndt_max_iter_ = this->declare_parameter<int>("max_iter", FALLBACK_NDT_ITER);
    ndt_miss_cnt_ = this->declare_parameter<int>("miss_cnt", FALLBACK_NDT_MISS_CNT);
    ndt_score_thresh_ = this->declare_parameter<double>("score_threshold", ROLLBACK_NDT_SCORE_THRESH);
    ndt_rollback_score_cnt_ = this->declare_parameter<int>("rollback_score_n", ROLLBACK_NDT_SCORE_CNT);
    ndt_fallback_score_cnt_ = this->declare_parameter<int>("fallback_score_n", FALLBACK_NDT_SCORE_CNT);
    ndt_reset_score_cnt_ = this->declare_parameter<int>("reset_score_n", NDT_RESET_CNT);


    double gnss_valid_duration_sec =
      this->declare_parameter<double>("gnss_valid_duration", GNSS_VALID);

    gnss_valid_duration_ =
      std::make_shared<rclcpp::Duration>(rclcpp::Duration::from_seconds(gnss_valid_duration_sec));

    // iter_topic_ = this->declare_parameter<std::string>(
    //   "iter_topic",
    //   "/localization/pose_estimator/iteration_num");
    gnss_topic_ = this->declare_parameter<std::string>(
      "gnss_topic",
      "/sensing/gnss/pose_with_covariance");

    score_topic_ = this->declare_parameter<std::string>(
      "score_topic",
      "/localization/pose_estimator/transform_probability");
    backup_mode_topic_ = this->declare_parameter<std::string>(
      "backup_mode_topic",
      "/backup_mode");

    // LOG
    RCLCPP_INFO(
      this->get_logger(),
      "Supervisor params: max_iter=%d, miss_cnt=%d, score_th=%.3f, N=%d",
      ndt_max_iter_, ndt_miss_cnt_, ndt_score_thresh_, ndt_rollback_score_cnt_);

    // --- Publisher ---
    backup_mode_pub_ = this->create_publisher<std_msgs::msg::Bool>(backup_mode_topic_, 1);

    // --- Subscribers ---
    gnss_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      gnss_topic_, rclcpp::QoS(10),std::bind(&SupervisorNode::onGnss, this, std::placeholders::_1));

    score_sub_ = this->create_subscription<tier4_debug_msgs::msg::Float32Stamped>(
      score_topic_, rclcpp::QoS(10),
      std::bind(&SupervisorNode::onTransformProb, this, std::placeholders::_1));

    // --- Service ---
    ndt_trigger_service_name_ = this->declare_parameter<std::string>(
      "ndt_trigger_service",
      "/localization/pose_estimator/trigger_node"
    );

    ndt_trigger_client_ =
      this->create_client<std_srvs::srv::SetBool>(ndt_trigger_service_name_);


    // initial state publish
    publishBackupMode();
  }

private:
  enum class Mode { NORMAL = 0, BACKUP = 1 };

  // === Callbacks ===
  void onTransformProb(const tier4_debug_msgs::msg::Float32Stamped::SharedPtr msg)
  {
    const double score = msg->data;

    // rocovery
    bool good = (score >= ndt_score_thresh_);

    if (good) {
      consecutive_good_score_cnt_++;
      consecutive_bad_score_cnt_ = 0;
    } else {
      consecutive_good_score_cnt_ = 0;
      consecutive_bad_score_cnt_++;
    }
    // LOG
    // RCLCPP_INFO(
    //     this->get_logger(),
    //     "[Supervisor] score (score %.3f)",
    //     score);
    
    // NORMAL -> BACUP transition
    if (mode_ == Mode::NORMAL && consecutive_bad_score_cnt_ >= ndt_fallback_score_cnt_) {
      RCLCPP_INFO(
        this->get_logger(),
        "[Supervisor] Detected physical error (bad score %d for %d times). "
        "Switching to BACKUP.",
        consecutive_bad_score_cnt_, ndt_fallback_score_cnt_);
      switchToBackup();
    }

    // BACKUP -> NORMAL transition
    // 1. ndt reset
    if (mode_ == Mode::BACKUP && !ndt_reset_done_ && consecutive_good_score_cnt_ >= ndt_reset_score_cnt_) {
      // RCLCPP_INFO(
      //   this->get_logger(),
      //   "[Supervisor] Requesting NDT reset.");
      
      // if (callNdtTrigger(true)) {
      //   ndt_reset_done_ = true;
      // } else {
      //   RCLCPP_WARN(
      //   this->get_logger(),
      //   "[Supervisor] NDT reset request failed.");
      // }
    }
    
    // 2. rollback
    if (mode_ == Mode::BACKUP && consecutive_good_score_cnt_ >= ndt_rollback_score_cnt_) {
      RCLCPP_INFO(
        this->get_logger(),
        "[Supervisor] Recover from physical error (score >= %.3f, iter < %d for %d times). "
        "Switching back to NORMAL.",
        ndt_score_thresh_, ndt_max_iter_, consecutive_good_score_cnt_);
      switchToNormal();
    }
  }

  void onGnss(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    last_gnss_msg_ = msg;
    latest_gnss_stamp_ = rclcpp::Time(msg->header.stamp);
    has_gnss_pose_ = true;
  }



  // === Mode switching helpers ===
  void switchToBackup()
  {
    if (mode_ == Mode::BACKUP) {
      return;
    }
    mode_ = Mode::BACKUP;
    // consecutive_good_score_cnt_ = 0; 
    consecutive_bad_score_cnt_ = 0;
    ndt_reset_done_ = false;
    publishBackupMode();
  }

  void switchToNormal()
  {
    if (mode_ == Mode::NORMAL) {
      return;
    }
    mode_ = Mode::NORMAL;
    consecutive_max_iter_cnt_ = 0;
    consecutive_good_score_cnt_ = 0;
    publishBackupMode();
  }

  void publishBackupMode()
  {
    std_msgs::msg::Bool msg;
    msg.data = (mode_ == Mode::BACKUP);
    backup_mode_pub_->publish(msg);

    RCLCPP_INFO(
      this->get_logger(),
      "[Supervisor] Mode = %s",
      (mode_ == Mode::BACKUP ? "BACKUP" : "NORMAL"));
  }

  bool callNdtTrigger(bool flag)
  {
    if (!ndt_trigger_client_) {
      RCLCPP_WARN(this->get_logger(), "NDT trigger client is not created");
      return false;
    }

    if (!ndt_trigger_client_->service_is_ready()) {
      RCLCPP_WARN(
        this->get_logger(),
        "NDT trigger service (%s) is not available",
        ndt_trigger_service_name_.c_str());
      return false;
    }

    auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
    req->data = flag;

    std::string command_name = flag ? "Activation" : "Deactivation";
    
    RCLCPP_INFO(
        this->get_logger(),
        "NDT trigger send ready");

    auto future = ndt_trigger_client_->async_send_request(req);
    auto resp = future.get();

    RCLCPP_INFO(
        this->get_logger(),
        "NDT trigger sent");

    if (resp->success) {
      RCLCPP_INFO(
        this->get_logger(),
        "NDT %s succeeded: %s",
        command_name.c_str(), resp->message.c_str());
      return true;
    } else {
      RCLCPP_WARN(
        this->get_logger(),
        "NDT %s failed: %s",
        command_name.c_str(), resp->message.c_str());
      return false;
    }
  }


public:
  bool getGnssPose(geometry_msgs::msg::PoseWithCovarianceStamped & pose)
  {
    if (!has_gnss_pose_) {
      return false;
    }

    rclcpp::Time now = this->get_clock()->now();
    rclcpp::Duration elapsed = now - latest_gnss_stamp_;

    if (elapsed <= *gnss_valid_duration_) {
      pose = *last_gnss_msg_;
      return true;
    } else {
      return false;
    }
  }

private:
  // === Members ===
  Mode mode_ = Mode::NORMAL;

  // Parameters
  int ndt_max_iter_;
  int ndt_miss_cnt_;
  double ndt_score_thresh_;
  int ndt_rollback_score_cnt_;
  int ndt_fallback_score_cnt_;
  int ndt_reset_score_cnt_;

  // std::string iter_topic_;
  std::string score_topic_;
  std::string backup_mode_topic_;
  std::string gnss_topic_;

  // State
  int last_iteration_num_ = 0;
  int consecutive_max_iter_cnt_ = 0;
  int consecutive_good_score_cnt_ = 0;
  int consecutive_bad_score_cnt_ = 0;
  bool ndt_reset_done_ = false;

  // NDT trigger service
  std::string ndt_trigger_service_name_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr ndt_trigger_client_;

  // GNSS
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr last_gnss_msg_;
  rclcpp::Time latest_gnss_stamp_;
  bool has_gnss_pose_ = false;
  std::shared_ptr<rclcpp::Duration> gnss_valid_duration_;

  // ROS
  // rclcpp::Subscription<tier4_debug_msgs::msg::Int32Stamped>::SharedPtr iter_sub_;
  rclcpp::Subscription<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr score_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr backup_mode_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr gnss_sub_;
};


// === main ===
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SupervisorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
