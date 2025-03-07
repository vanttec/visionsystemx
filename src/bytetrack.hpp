#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "usv_interfaces/msg/zbbox.hpp"
#include "usv_interfaces/msg/zbbox_array.hpp"

namespace bytetrack {

// Kalman Filter state representation
// [x, y, w, h, vx, vy, vw, vh]
// x, y: center position
// w, h: width and height
// vx, vy, vw, vh: respective velocities
typedef Eigen::Matrix<float, 8, 1> StateVector;
typedef Eigen::Matrix<float, 4, 1> MeasurementVector;  // [x, y, w, h]

// Track status enumeration
enum TrackState { New = 0, Tracked, Lost, Removed };

class KalmanFilter {
public:
    KalmanFilter() {
        // State transition matrix
        F_ = Eigen::Matrix<float, 8, 8>::Identity();
        // For positions and dimensions: add velocity
        for (int i = 0; i < 4; ++i) {
            F_(i, i + 4) = 1.0;
        }
        
        // Measurement matrix (we only observe position and size)
        H_ = Eigen::Matrix<float, 4, 8>::Zero();
        for (int i = 0; i < 4; ++i) {
            H_(i, i) = 1.0;
        }
        
        // Process noise covariance
        Q_ = Eigen::Matrix<float, 8, 8>::Identity() * 0.1;
        // Higher noise for velocity components
        for (int i = 4; i < 8; ++i) {
            Q_(i, i) = 0.2;
        }
        
        // Measurement noise covariance
        R_ = Eigen::Matrix<float, 4, 4>::Identity() * 1.0;
        
        // Initial state covariance
        P_ = Eigen::Matrix<float, 8, 8>::Identity() * 10.0;
    }
    
    void initiate(const MeasurementVector& measurement) {
        // Initialize state: [x, y, w, h, 0, 0, 0, 0]
        state_ = StateVector::Zero();
        state_.head(4) = measurement;
        
        // Initialize covariance
        covariance_ = P_;
    }
    
    void predict() {
        // x = F * x
        state_ = F_ * state_;
        
        // P = F * P * F^T + Q
        covariance_ = F_ * covariance_ * F_.transpose() + Q_;
    }
    
    void update(const MeasurementVector& measurement) {
        // y = z - H * x
        Eigen::Matrix<float, 4, 1> y = measurement - H_ * state_;
        
        // S = H * P * H^T + R
        Eigen::Matrix<float, 4, 4> S = H_ * covariance_ * H_.transpose() + R_;
        
        // K = P * H^T * S^-1
        Eigen::Matrix<float, 8, 4> K = covariance_ * H_.transpose() * S.inverse();
        
        // x = x + K * y
        state_ = state_ + K * y;
        
        // P = (I - K * H) * P
        covariance_ = (Eigen::Matrix<float, 8, 8>::Identity() - K * H_) * covariance_;
    }
    
    StateVector getState() const {
        return state_;
    }
    
    MeasurementVector getMeasurement() const {
        return H_ * state_;
    }

private:
    StateVector state_;
    Eigen::Matrix<float, 8, 8> covariance_;
    Eigen::Matrix<float, 8, 8> F_;  // State transition matrix
    Eigen::Matrix<float, 4, 8> H_;  // Measurement matrix
    Eigen::Matrix<float, 8, 8> Q_;  // Process noise covariance
    Eigen::Matrix<float, 4, 4> R_;  // Measurement noise covariance
    Eigen::Matrix<float, 8, 8> P_;  // Initial state covariance
};

// Single Object Track
class STrack {
public:
    STrack(const usv_interfaces::msg::Zbbox& detection, int frame_id)
        : detection_(detection), 
          track_id_(-1), 
          state_(TrackState::New), 
          frame_id_(frame_id),
          start_frame_(frame_id),
          tracklet_len_(0) {
        
        // Convert detection to measurement vector
        float x = (detection.x0 + detection.x1) * 0.5f;
        float y = (detection.y0 + detection.y1) * 0.5f;
        float w = detection.x1 - detection.x0;
        float h = detection.y1 - detection.y0;
        
        MeasurementVector measurement;
        measurement << x, y, w, h;
        
        // Initialize Kalman filter
        kalman_filter_.initiate(measurement);
    }
    
    void predict() {
        kalman_filter_.predict();
    }
    
    void update(const STrack& new_track, int frame_id) {
        frame_id_ = frame_id;
        tracklet_len_++;
        
        state_ = TrackState::Tracked;
        is_activated_ = true;
        
        detection_ = new_track.detection_;
        
        // Convert detection to measurement
        float x = (detection_.x0 + detection_.x1) * 0.5f;
        float y = (detection_.y0 + detection_.y1) * 0.5f;
        float w = detection_.x1 - detection_.x0;
        float h = detection_.y1 - detection_.y0;
        
        MeasurementVector measurement;
        measurement << x, y, w, h;
        
        // Update Kalman filter
        kalman_filter_.update(measurement);
    }
    
    void markAsLost() {
        state_ = TrackState::Lost;
    }
    
    void markAsRemoved() {
        state_ = TrackState::Removed;
    }
    
    bool isActivated() const {
        return is_activated_;
    }
    
    void activate(int track_id, int frame_id) {
        track_id_ = track_id;
        state_ = TrackState::Tracked;
        is_activated_ = true;
        frame_id_ = frame_id;
        
        // If there's a UUID in the detection, keep it; otherwise generate one
        if (detection_.uuid.empty()) {
            detection_.uuid = "track_" + std::to_string(track_id);
        }
    }
    
    int getTrackId() const {
        return track_id_;
    }
    
    TrackState getState() const {
        return state_;
    }
    
    usv_interfaces::msg::Zbbox getDetection() const {
        // If the track has been updated by the Kalman filter, update the bounding box
        if (state_ == TrackState::Tracked) {
            MeasurementVector state = kalman_filter_.getMeasurement();
            float x = state(0);
            float y = state(1);
            float w = state(2);
            float h = state(3);
            
            usv_interfaces::msg::Zbbox updated = detection_;
            updated.x0 = x - w/2;
            updated.y0 = y - h/2;
            updated.x1 = x + w/2;
            updated.y1 = y + h/2;
            
            return updated;
        }
        
        return detection_;
    }
    
    int getLastFrameId() const {
        return frame_id_;
    }
    
    int getStartFrame() const {
        return start_frame_;
    }
    
    int getTrackletLength() const {
        return tracklet_len_;
    }
    
    static float calculateIoU(const STrack& st1, const STrack& st2) {
        auto det1 = st1.getDetection();
        auto det2 = st2.getDetection();
        
        float x1 = std::max(det1.x0, det2.x0);
        float y1 = std::max(det1.y0, det2.y0);
        float x2 = std::min(det1.x1, det2.x1);
        float y2 = std::min(det1.y1, det2.y1);
        
        float w = std::max(0.0f, x2 - x1);
        float h = std::max(0.0f, y2 - y1);
        float intersection = w * h;
        
        float area1 = (det1.x1 - det1.x0) * (det1.y1 - det1.y0);
        float area2 = (det2.x1 - det2.x0) * (det2.y1 - det2.y0);
        float union_area = area1 + area2 - intersection;
        
        if (union_area > 0) {
            return intersection / union_area;
        }
        return 0;
    }
    
private:
    usv_interfaces::msg::Zbbox detection_;
    int track_id_;
    TrackState state_;
    bool is_activated_ = false;
    int frame_id_;
    int start_frame_;
    int tracklet_len_;
    KalmanFilter kalman_filter_;
};

// Implementation of the Hungarian algorithm for assignment problems
class HungarianAlgorithm {
public:
    // Solve the assignment problem given a cost matrix
    // Returns a vector of column indices for each row
    std::vector<int> solve(const std::vector<std::vector<float>>& cost_matrix) {
        if (cost_matrix.empty()) {
            return {};
        }
        
        int rows = cost_matrix.size();
        int cols = cost_matrix[0].size();
        
        // Pad the cost matrix to make it square if necessary
        int size = std::max(rows, cols);
        std::vector<std::vector<float>> padded_cost(size, std::vector<float>(size, 0));
        
        // Fill the padded cost matrix
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                padded_cost[i][j] = cost_matrix[i][j];
            }
        }
        
        // For rows > cols, fill with large values (impossible assignments)
        for (int i = rows; i < size; i++) {
            for (int j = 0; j < size; j++) {
                padded_cost[i][j] = 1e9;
            }
        }
        
        // For cols > rows, fill with large values (impossible assignments)
        for (int i = 0; i < rows; i++) {
            for (int j = cols; j < size; j++) {
                padded_cost[i][j] = 1e9;
            }
        }
        
        // Step 1: Subtract the smallest element in each row from all elements in that row
        for (int i = 0; i < size; i++) {
            float min_val = *std::min_element(padded_cost[i].begin(), padded_cost[i].end());
            for (int j = 0; j < size; j++) {
                padded_cost[i][j] -= min_val;
            }
        }
        
        // Step 2: Subtract the smallest element in each column from all elements in that column
        for (int j = 0; j < size; j++) {
            float min_val = padded_cost[0][j];
            for (int i = 1; i < size; i++) {
                min_val = std::min(min_val, padded_cost[i][j]);
            }
            for (int i = 0; i < size; i++) {
                padded_cost[i][j] -= min_val;
            }
        }
        
        // Step 3: Cover all zeros using minimum number of lines
        std::vector<bool> row_covered(size, false);
        std::vector<bool> col_covered(size, false);
        std::vector<int> row_assignments(size, -1);
        std::vector<int> col_assignments(size, -1);
        
        // Initial assignment: Greedy algorithm
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (padded_cost[i][j] == 0 && !row_covered[i] && !col_covered[j]) {
                    row_assignments[i] = j;
                    col_assignments[j] = i;
                    row_covered[i] = true;
                    col_covered[j] = true;
                }
            }
        }
        
        // Reset covered vectors for the algorithm
        std::fill(row_covered.begin(), row_covered.end(), false);
        std::fill(col_covered.begin(), col_covered.end(), false);
        
        // Mark rows with no assignments
        for (int i = 0; i < size; i++) {
            if (row_assignments[i] == -1) {
                row_covered[i] = true;
            }
        }
        
        // Iteratively find better assignments
        while (true) {
            // Find uncovered zeros and try to make assignments
            bool found_uncovered_zero = false;
            
            for (int i = 0; i < size; i++) {
                if (row_covered[i]) continue;
                
                for (int j = 0; j < size; j++) {
                    if (col_covered[j]) continue;
                    
                    if (padded_cost[i][j] == 0) {
                        // Found an uncovered zero
                        row_assignments[i] = j;
                        col_assignments[j] = i;
                        
                        // Mark as covered
                        row_covered[i] = true;
                        col_covered[j] = true;
                        
                        found_uncovered_zero = true;
                        break;
                    }
                }
            }
            
            if (!found_uncovered_zero) {
                break;
            }
        }
        
        // Check if we have a complete assignment
        int num_assigned = 0;
        for (int i = 0; i < size; i++) {
            if (row_assignments[i] != -1 && row_assignments[i] < cols) {
                num_assigned++;
            }
        }
        
        if (num_assigned == rows) {
            // We found a complete assignment
            std::vector<int> result;
            for (int i = 0; i < rows; i++) {
                result.push_back(row_assignments[i]);
            }
            return result;
        }
        
        // If we don't have a complete assignment, return default assignments
        std::vector<int> result(rows, -1);
        for (int i = 0; i < rows; i++) {
            // Check if we have a valid assignment for this row
            for (int j = 0; j < cols; j++) {
                if (cost_matrix[i][j] < 1e6) {  // Not a "forbidden" assignment
                    result[i] = j;
                    break;
                }
            }
        }
        
        return result;
    }
};

// ByteTrack implementation
class ByteTrack {
public:
    ByteTrack(float high_thresh = 0.6, 
             float low_thresh = 0.1, 
             int max_time_lost = 30,
             int min_hits_to_activate = 3,
             rclcpp::Logger logger = rclcpp::get_logger("bytetrack"))
        : high_threshold_(high_thresh),
          low_threshold_(low_thresh),
          max_time_lost_(max_time_lost),
          min_hits_to_activate_(min_hits_to_activate),
          frame_id_(0),
          next_track_id_(0),
          logger_(logger) {
        RCLCPP_INFO(logger_, "ByteTrack initialized with high_thresh=%.2f, low_thresh=%.2f, max_time_lost=%d, min_hits=%d",
                 high_thresh, low_thresh, max_time_lost, min_hits_to_activate);
    }
    
    // Update tracker parameters dynamically
    void updateParameters(float high_thresh, float low_thresh, int max_time_lost, int min_hits_to_activate) {
        if (high_threshold_ != high_thresh || 
            low_threshold_ != low_thresh || 
            max_time_lost_ != max_time_lost || 
            min_hits_to_activate_ != min_hits_to_activate) {
            
            RCLCPP_INFO(logger_, "ByteTrack parameters updated: high_thresh=%.2f, low_thresh=%.2f, max_time_lost=%d, min_hits=%d",
                     high_thresh, low_thresh, max_time_lost, min_hits_to_activate);
                     
            high_threshold_ = high_thresh;
            low_threshold_ = low_thresh;
            max_time_lost_ = max_time_lost;
            min_hits_to_activate_ = min_hits_to_activate;
        }
    }
    
    // Update tracks with new detections
    usv_interfaces::msg::ZbboxArray update(const usv_interfaces::msg::ZbboxArray& detections) {
        frame_id_++;
        
        // Step 1: Get new detections and update existing tracks
        std::vector<STrack> detections_high;
        std::vector<STrack> detections_low;
        
        // Separate detections by confidence
        for (const auto& det : detections.boxes) {
            if (det.prob >= high_threshold_) {
                detections_high.emplace_back(det, frame_id_);
            } else if (det.prob >= low_threshold_) {
                detections_low.emplace_back(det, frame_id_);
            }
        }
        
        // Predict new locations of existing tracks
        for (auto& track : tracked_tracks_) {
            track.predict();
        }
        for (auto& track : lost_tracks_) {
            track.predict();
        }
        
        // Step 2: Associate high confidence detections with tracked tracks
        std::vector<STrack> unmatched_tracks;
        std::vector<STrack> unmatched_detections;
        matchDetections(tracked_tracks_, detections_high, unmatched_tracks, unmatched_detections);
        
        // Step 3: Associate remaining tracks with low confidence detections
        std::vector<STrack> unmatched_tracks_second;
        std::vector<STrack> unmatched_detections_second;
        matchDetections(unmatched_tracks, detections_low, unmatched_tracks_second, unmatched_detections_second);
        
        // Step 4: Process unmatched tracks and detections
        for (auto& track : unmatched_tracks_second) {
            if (track.getState() == TrackState::Tracked) {
                if (frame_id_ - track.getLastFrameId() > max_time_lost_) {
                    track.markAsRemoved();
                    removed_tracks_.push_back(track);
                } else {
                    track.markAsLost();
                    lost_tracks_.push_back(track);
                }
            }
        }
        
        // Step 5: Initialize new tracks from remaining high-confidence detections
        for (auto& detection : unmatched_detections) {
            if (detection.getDetection().prob >= high_threshold_) {
                STrack new_track = detection;
                new_track.activate(next_track_id_++, frame_id_);
                
                // If the track has been active for min_hits_to_activate_ frames, activate it
                if (new_track.getTrackletLength() >= min_hits_to_activate_) {
                    activated_tracks_.push_back(new_track);
                } else {
                    // Otherwise, keep it as a tentative track
                    unconfirmed_tracks_.push_back(new_track);
                }
            }
        }
        
        // Step 6: Update track lists
        std::vector<STrack> new_tracked_tracks;
        
        for (auto& track : activated_tracks_) {
            new_tracked_tracks.push_back(track);
        }
        
        for (auto& track : unconfirmed_tracks_) {
            if (track.getState() == TrackState::Tracked) {
                new_tracked_tracks.push_back(track);
            }
        }
        
        tracked_tracks_ = new_tracked_tracks;
        
        // Step 7: Prepare output - include all tracked and recently lost tracks
        usv_interfaces::msg::ZbboxArray result;
        result.header = detections.header;
        
        for (const auto& track : tracked_tracks_) {
            if (track.isActivated()) {
                result.boxes.push_back(track.getDetection());
            }
        }
        
        // Also include recently lost tracks (helpful for tracking continuity)
        for (const auto& track : lost_tracks_) {
            if (frame_id_ - track.getLastFrameId() <= 1) {  // Include only very recently lost tracks
                auto det = track.getDetection();
                det.prob *= 0.9;  // Reduce confidence slightly
                result.boxes.push_back(det);
            }
        }
        
        // Debug output
        RCLCPP_DEBUG(logger_, "Frame %d: %zu tracked, %zu lost, %zu removed",
                 frame_id_, tracked_tracks_.size(), lost_tracks_.size(), removed_tracks_.size());
        
        // Clear temporary vectors for next frame
        activated_tracks_.clear();
        unconfirmed_tracks_.clear();
        
        // Remove stale tracks from lost_tracks_
        std::vector<STrack> new_lost_tracks;
        for (const auto& track : lost_tracks_) {
            if (frame_id_ - track.getLastFrameId() <= max_time_lost_) {
                new_lost_tracks.push_back(track);
            }
        }
        lost_tracks_ = new_lost_tracks;
        
        return result;
    }
    
private:
    void matchDetections(std::vector<STrack>& tracks, 
                         const std::vector<STrack>& detections,
                         std::vector<STrack>& unmatched_tracks,
                         std::vector<STrack>& unmatched_detections) {
        if (tracks.empty() || detections.empty()) {
            unmatched_tracks = tracks;
            unmatched_detections = detections;
            return;
        }
        
        // Compute IoU matrix
        std::vector<std::vector<float>> iou_matrix(tracks.size(), std::vector<float>(detections.size()));
        
        for (size_t i = 0; i < tracks.size(); i++) {
            for (size_t j = 0; j < detections.size(); j++) {
                iou_matrix[i][j] = 1.0f - STrack::calculateIoU(tracks[i], detections[j]);
            }
        }
        
        // Apply Hungarian algorithm to find optimal matches
        HungarianAlgorithm hungarian;
        std::vector<int> assignments = hungarian.solve(iou_matrix);
        
        // Process matches and unmatched tracks/detections
        std::set<int> unmatched_detection_indices;
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_detection_indices.insert(i);
        }
        
        std::vector<STrack> matched_tracks;
        for (size_t i = 0; i < assignments.size(); i++) {
            int detection_idx = assignments[i];
            
            if (detection_idx != -1) {
                // We have a match, check if it's a valid match (IoU > 0)
                if (iou_matrix[i][detection_idx] <= 0.8f) { // IoU threshold for valid match
                    tracks[i].update(detections[detection_idx], frame_id_);
                    matched_tracks.push_back(tracks[i]);
                    unmatched_detection_indices.erase(detection_idx);
                } else {
                    // Invalid match, add to unmatched lists
                    unmatched_tracks.push_back(tracks[i]);
                }
            } else {
                // No match found for this track
                unmatched_tracks.push_back(tracks[i]);
            }
        }
        
        // Add unmatched detections
        for (int idx : unmatched_detection_indices) {
            unmatched_detections.push_back(detections[idx]);
        }
        
        // Update tracks list to only include matched tracks
        tracks = matched_tracks;
    }
    
    // Thresholds for detection confidence
    float high_threshold_;
    float low_threshold_;
    
    // Maximum time (in frames) a track can be lost before being removed
    int max_time_lost_;
    
    // Minimum hits required to activate a track
    int min_hits_to_activate_;
    
    // Current frame ID
    int frame_id_;
    
    // Next available track ID
    int next_track_id_;
    
    // Lists of tracks in different states
    std::vector<STrack> tracked_tracks_;     // Currently tracked
    std::vector<STrack> lost_tracks_;        // Lost but may reappear
    std::vector<STrack> removed_tracks_;     // Permanently removed
    
    // Temporary track lists for processing
    std::vector<STrack> activated_tracks_;    // New tracks being activated this frame
    std::vector<STrack> unconfirmed_tracks_;  // Potential tracks not yet fully confirmed
    
    // Logger
    rclcpp::Logger logger_;
};

} // namespace bytetrack