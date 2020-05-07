#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_ << 0,
        0,
        0,
        0,
        0;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // set weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = weight;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // Input:
  // x_ :: [5 x 1]
  // P_ :: [5 x 5]

  // Output:
  // x_ :: [5 x 1]
  // P_ :: [5 x 5]
  // Xsig_pred_ :: [5 x 15]

  // ----------------------------
  //    GENERATE SIGMA POINTS
  // ----------------------------

  // create augmented mean state vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.setZero();
  x_aug.head(n_x_) = x_;
  
  // process noise has zero mean
  x_aug(n_x_ + 0) = 0;
  x_aug(n_x_ + 1) = 0;

  // create augmented state covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  // process noise covariance
  P_aug(n_x_ + 0, n_x_ + 0) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // calculate square root of P_aug
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
}

  // ----------------------------
  //    PREDICT SIGMA POINTS
  // ----------------------------

  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px        = Xsig_aug(0, i);
    double py        = Xsig_aug(1, i);
    double v         = Xsig_aug(2, i);
    double psi       = Xsig_aug(3, i);
    double psi_d     = Xsig_aug(4, i);
    double nu_a      = Xsig_aug(5, i);
    double nu_psi_dd = Xsig_aug(6, i);

    VectorXd x = VectorXd(n_x_);
    x << px,
         py,
         v,
         psi,
         psi_d;

    VectorXd s1 = VectorXd(n_x_);
    double atol = 1e-3;
    if (fabs(psi_d) > atol) {
      s1 << v / psi_d * (sin(psi + psi_d * delta_t) - sin(psi)),
            v / psi_d * (-cos(psi + psi_d * delta_t) + cos(psi)),
            0,
            psi_d * delta_t,
            0;
    } else {
      // avoid division by zero
      s1 << v * cos(psi) * delta_t,
            v * sin(psi) * delta_t,
            0,
            psi_d * delta_t,
            0;
    }

    VectorXd s2 = VectorXd(n_x_);
    s2 << 0.5 * (delta_t * delta_t) * cos(psi) * nu_a,
          0.5 * (delta_t * delta_t) * sin(psi) * nu_a,
          delta_t * nu_a,
          0.5 * (delta_t * delta_t) * nu_psi_dd,
          delta_t * nu_psi_dd;

    Xsig_pred_.col(i) = x + s1 + s2;
  }

  // ----------------------------
  // PREDICT MEAN AND COVARIANCE
  // ----------------------------

  // predict state mean
  x_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // normalize (yaw) angle to (-pi, pi)
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // Input:
  // z  :: [2 x 1]
  // P_ :: [5 x 5]
  // Xsig_pred_ :: [5 x 15]

  // Output:
  // x_ :: [5 x 1]
  // P_ :: [5 x 5]

  // set measurement dimension, lidar can measure px and py
  int n_z = 2;

  // incoming lidar measurement
  VectorXd z = meas_package.raw_measurements_;

  // ----------------------------
  //     PREDICT MEASUREMENT
  // ----------------------------

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Input
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    // Measurement Model
    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // calculate mean predicted measurement
  z_pred.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R.setZero();
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;

  S.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
}
  S += R;

  // ----------------------------
  //       UPDATE STATE
  // ----------------------------

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // normalize (yaw) angle to (-pi, pi)
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}