#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <fstream>
#include <sstream>
#include <cmath>

// TODO: 替换为你实际的球状态消息类型
// #include "motor_control_ros2/msg/ball_state.hpp"

namespace strike_learning {

class StrikeDataCollectorNode : public rclcpp::Node {
public:
    StrikeDataCollectorNode() : Node("strike_data_collector_node") {
        RCLCPP_INFO(this->get_logger(), "Strike Data Collector Node started");
        
        // 初始化 CSV 文件
        csv_filename_ = "strike_data.csv";
        csv_file_.open(csv_filename_, std::ios::app);  // 追加模式
        if (!csv_file_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_filename_.c_str());
            throw std::runtime_error("Cannot open CSV file");
        }
        
        // 如果文件为空，写入表头
        csv_file_.seekp(0, std::ios::end);
        if (csv_file_.tellp() == 0) {
            csv_file_ << "timestamp,x_robot,y_robot,theta_robot,x_ball,y_ball,vx_ball,vy_ball,strike_angle,strike_force\n";
        }
        
        // 订阅 /odom (车的位姿)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                this->odom_callback(msg);
            });
        
        // TODO: 订阅 /ball_tracker (球的状态)
        // ball_sub_ = this->create_subscription<BallState>(
        //     "/ball_tracker", 10,
        //     [this](const BallState::SharedPtr msg) {
        //         this->ball_callback(msg);
        //     });
        
        // TODO: 订阅手柄控制的击球命令 (ground truth)
        // strike_cmd_sub_ = this->create_subscription<StrikeCommand>(
        //     "/manual_strike_command", 10,
        //     [this](const StrikeCommand::SharedPtr msg) {
        //         this->strike_callback(msg);
        //     });
        
        // 定时器：定期打印状态
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            [this]() { this->print_status(); });
        
        RCLCPP_INFO(this->get_logger(), "Subscribed to /odom");
        RCLCPP_INFO(this->get_logger(), "CSV file: %s", csv_filename_.c_str());
    }
    
    ~StrikeDataCollectorNode() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }

private:
    // 回调：接收机器人位姿
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        
        // 从四元数提取偏航角 (theta)
        auto& q = msg->pose.pose.orientation;
        double theta = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        );
        
        robot_x_ = x;
        robot_y_ = y;
        robot_theta_ = theta;
        
        // 第一次打印
        static bool first = true;
        if (first) {
            RCLCPP_INFO(this->get_logger(), "First odom received: x=%.2f, y=%.2f, theta=%.2f", x, y, theta);
            first = false;
        }
    }
    
    // TODO: 实现球状态回调
    // void ball_callback(const BallState::SharedPtr msg) {
    //     ball_x_ = msg->position.x;
    //     ball_y_ = msg->position.y;
    //     ball_vx_ = msg->velocity.x;
    //     ball_vy_ = msg->velocity.y;
    // }
    
    // TODO: 实现手柄击球命令回调（记录 ground truth）
    // void strike_callback(const StrikeCommand::SharedPtr msg) {
    //     strike_angle_ = msg->target_angle;
    //     strike_force_ = msg->target_force;
    //     
    //     // 写入 CSV
    //     record_data();
    // }
    
    // 记录一条数据
    void record_data() {
        if (!csv_file_.is_open()) return;
        
        auto now = this->now();
        csv_file_ << now.nanoseconds() / 1e9 << ","
                  << robot_x_ << "," << robot_y_ << "," << robot_theta_ << ","
                  << ball_x_ << "," << ball_y_ << "," << ball_vx_ << "," << ball_vy_ << ","
                  << strike_angle_ << "," << strike_force_ << "\n";
        csv_file_.flush();
        
        data_count_++;
    }
    
    // 打印状态信息
    void print_status() {
        RCLCPP_INFO(this->get_logger(),
            "Robot: (%.2f, %.2f, %.2f) | Ball: (%.2f, %.2f) v=(%.2f, %.2f) | Recorded: %d",
            robot_x_, robot_y_, robot_theta_,
            ball_x_, ball_y_, ball_vx_, ball_vy_,
            data_count_);
    }
    
    // 成员变量
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    // rclcpp::Subscription<BallState>::SharedPtr ball_sub_;
    // rclcpp::Subscription<StrikeCommand>::SharedPtr strike_cmd_sub_;
    
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    std::string csv_filename_;
    std::ofstream csv_file_;
    
    // 机器人位姿
    double robot_x_ = 0.0;
    double robot_y_ = 0.0;
    double robot_theta_ = 0.0;
    
    // 球的状态
    double ball_x_ = 0.0;
    double ball_y_ = 0.0;
    double ball_vx_ = 0.0;
    double ball_vy_ = 0.0;
    
    // 击球命令
    double strike_angle_ = 0.0;
    double strike_force_ = 0.0;
    
    // 统计
    int data_count_ = 0;
};

}  // namespace strike_learning

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<strike_learning::StrikeDataCollectorNode>());
    rclcpp::shutdown();
    return 0;
}
