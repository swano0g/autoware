#include <chrono>
#include <memory>
#include <future>

#include "rclcpp/rclcpp.hpp"

#include "std_srvs/srv/set_bool.hpp"

#include <tier4_debug_msgs/msg/float32_stamped.hpp>
#include <tier4_debug_msgs/msg/int32_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include "std_msgs/msg/bool.hpp"
#include <autoware_adapi_v1_msgs/srv/initialize_localization.hpp>
#include "std_msgs/msg/string.hpp"



using InitializeLocalization =
  autoware_adapi_v1_msgs::srv::InitializeLocalization;

using namespace std::chrono_literals;

// FALLBACK NORMAL -> BACKUP
#define FALLBACK_NDT_ITER           21
#define FALLBACK_NDT_MISS_CNT       1

#define FALLBACK_NDT_SCORE_CNT      1

// ROLLBACK BACKUP -> NORMAL
#define ROLLBACK_NDT_SCORE_THRESH   2.3
#define ROLLBACK_NDT_SCORE_CNT      50

#define ROLLBACK_NDT_SEMI_SCORE_THRESH 2.0
#define ROLLBACK_NDT_SEMI_SCORE_CNT    50


#define INIT_RETRY_INTERVAL_SEC 80

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
    

    iter_topic_ = this->declare_parameter<std::string>(
      "iter_topic",
      "/localization/pose_estimator/iteration_num");
    score_topic_ = this->declare_parameter<std::string>(
      "score_topic",
      "/localization/pose_estimator/nearest_voxel_transformation_likelihood");
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
    score_sub_ = this->create_subscription<tier4_debug_msgs::msg::Float32Stamped>(
      score_topic_, rclcpp::QoS(10),
      std::bind(&SupervisorNode::onLikelihood, this, std::placeholders::_1));

    iter_sub_ = this->create_subscription<tier4_debug_msgs::msg::Int32Stamped>(
      iter_topic_, rclcpp::QoS(10),
      std::bind(&SupervisorNode::onIterCallback, this, std::placeholders::_1));

    // --- Service ---
    ndt_trigger_service_name_ = this->declare_parameter<std::string>(
      "ndt_trigger_service",
      "/localization/pose_estimator/trigger_node"

    );

    localization_init_client_ =
      this->create_client<InitializeLocalization>("/localization/initialize");

    ndt_trigger_client_ =
      this->create_client<std_srvs::srv::SetBool>(ndt_trigger_service_name_);


    init_debug_topic_ = this->declare_parameter<std::string>(
      "init_debug_topic",
      "/localization/init_debug");
    init_debug_pub_ = this->create_publisher<std_msgs::msg::String>(init_debug_topic_, 10);


    // initial state publish
    publishBackupMode();
  }

private:
  enum class Mode { NORMAL = 0, BACKUP = 1 };

  // === Callbacks ===
  void onIterCallback(const tier4_debug_msgs::msg::Int32Stamped::SharedPtr msg)
  {
    const int iter = msg->data;
    last_iteration_num_ = iter;

    const bool hit_max_iter = (iter >= ndt_max_iter_);

    if (hit_max_iter) {
      consecutive_max_iter_cnt_++;
      consecutive_semi_good_score_cnt_ = 0;
      consecutive_good_score_cnt_ = 0;
    } else {
      consecutive_max_iter_cnt_ = 0;
    }

    if (mode_ == Mode::NORMAL && consecutive_max_iter_cnt_ >= ndt_miss_cnt_) {
      RCLCPP_WARN(
        this->get_logger(),
        "[Supervisor] Physical error detected! (iteration_num reached max_iter=%d for %d times)",
        ndt_max_iter_, consecutive_max_iter_cnt_);

      switchToBackup();
    }
  }



  void onLikelihood(const tier4_debug_msgs::msg::Float32Stamped::SharedPtr msg)
  {
    const double score = msg->data;

    const bool semi_good = (score >= ROLLBACK_NDT_SEMI_SCORE_THRESH);
    const bool good = (score >= ndt_score_thresh_);

    if (semi_good) {
      consecutive_semi_good_score_cnt_++;
    } else {
      consecutive_semi_good_score_cnt_ = 0;
    }

    if (good) {
      consecutive_good_score_cnt_++;
    } else {
      consecutive_good_score_cnt_ = 0;
    }


    if (mode_ == Mode::BACKUP && consecutive_good_score_cnt_ < 5) {
      if (last_localization_initialize_.nanoseconds() == 0) {
        last_localization_initialize_ = this->now();
      } else {
        const auto now = this->now();
        const double elapsed = (now - last_localization_initialize_).seconds();

        if (elapsed > INIT_RETRY_INTERVAL_SEC) {
          RCLCPP_INFO(this->get_logger(),
                      "last localization initialize elapsed: %.1f -> start initialization",
                      elapsed);

          callLocalizationInitializeWithGnss();
          last_localization_initialize_ = this->now();
        }
      }
    }


    if (mode_ == Mode::BACKUP && consecutive_good_score_cnt_ >= ndt_rollback_score_cnt_)
    {
      RCLCPP_INFO(
        this->get_logger(),
        "[Supervisor] Recover from physical error "
        "(score >= %.3f for %d times). Switching back to NORMAL.",
        ndt_score_thresh_, consecutive_good_score_cnt_);

      switchToNormal();
    }
  }



  // === Mode switching helpers ===
  void switchToBackup()
  {
    if (mode_ == Mode::BACKUP) {
      return;
    }
    mode_ = Mode::BACKUP;
    consecutive_good_score_cnt_ = 0; 
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


bool callLocalizationInitializeWithGnss()
{
  if (!localization_init_client_) {
    RCLCPP_WARN(this->get_logger(), "Localization init client is not created");
    return false;
  }

  if (!localization_init_client_->service_is_ready()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Localization init service (%s) is not available",
      "/localization/initialize");
    return false;
  }

  auto req = std::make_shared<InitializeLocalization::Request>();

  req->pose.clear();

  if (init_debug_pub_) {
    std_msgs::msg::String dbg;
    dbg.data = "[Supervisor] init requested (GNSS-based)";
    init_debug_pub_->publish(dbg);
  }


  RCLCPP_INFO(this->get_logger(), "Sending localization initialize (GNSS-based)");

  auto future = localization_init_client_->async_send_request(
    req,
    [this](rclcpp::Client<InitializeLocalization>::SharedFuture result_future) {
      try {
        auto resp = result_future.get();
        const auto & status = resp->status;
        std_msgs::msg::String dbg;

        if (status.success) {
          RCLCPP_INFO(
            this->get_logger(),
            "Localization initialize succeeded: code=%u, msg=\"%s\"",
            status.code, status.message.c_str());
            dbg.data = "[Supervisor] init succeeded: code=" +
              std::to_string(status.code) + " msg=" + status.message;
        } else {
          RCLCPP_WARN(
            this->get_logger(),
            "Localization initialize failed: code=%u, msg=\"%s\"",
            status.code, status.message.c_str());
            dbg.data = "[Supervisor] init FAILED: code=" +
              std::to_string(status.code) + " msg=" + status.message;
        }
        if (init_debug_pub_) {
          init_debug_pub_->publish(dbg);
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(),
                     "Exception in localization init response callback: %s",
                     e.what());
      }
    });

  (void)future;
  return true;
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

  std::string iter_topic_;
  std::string score_topic_;
  std::string backup_mode_topic_;

  // State
  int last_iteration_num_ = 0;
  int consecutive_max_iter_cnt_ = 0;
  int consecutive_good_score_cnt_ = 0;
  int consecutive_semi_good_score_cnt_ = 0;

  // NDT trigger service
  std::string ndt_trigger_service_name_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr ndt_trigger_client_;
  rclcpp::Client<InitializeLocalization>::SharedPtr localization_init_client_;

  // init timer
  rclcpp::Time last_localization_initialize_{0, 0, RCL_ROS_TIME};

  // ROS
  rclcpp::Subscription<tier4_debug_msgs::msg::Int32Stamped>::SharedPtr iter_sub_;
  rclcpp::Subscription<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr score_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr backup_mode_pub_;

  //debug
  std::string init_debug_topic_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr init_debug_pub_;
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
