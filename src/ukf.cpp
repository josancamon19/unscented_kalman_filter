#include <utility>
#include <vector>

#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

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

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.12;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.14;

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
     * Hint: one or more values initialized above might be wildly off... (stda, std_yawdd_)
     */
    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_aug_;
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage m) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */
    // Init P
    if (!is_initialized_) {
        // Init x
        x_ << 0, 0, 1, 1, 1;
        P_ = MatrixXd::Identity(5, 5);

        if (m.sensor_type_ == MeasurementPackage::SensorType::LASER) {
            if (!use_laser_) return;
            x_(0) = m.raw_measurements_(0);
            x_(1) = m.raw_measurements_(1);
        } else {
            if (!use_radar_)return;
            x_(0) = cos(m.raw_measurements_(1)) * m.raw_measurements_(0);
            x_(1) = cos(m.raw_measurements_(0)) * m.raw_measurements_(0);
        }
        time_us_ = m.timestamp_;
        is_initialized_ = true;
        return;
    }


    float dt = (m.timestamp_ - time_us_) / 1000000.0;
    time_us_ = m.timestamp_;

    Prediction(dt);
    if (m.sensor_type_ == MeasurementPackage::SensorType::LASER) {
        if (!use_laser_) return;
        UpdateLidar(m);
    } else {
        if (!use_radar_) return;
        UpdateRadar(m);
    }

}

void UKF::Prediction(double delta_t) {
    /**  DONE
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */
    MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
    AugmentedSigmaPoints(&Xsig_aug);

    Xsig_pred_ = PredictSigmaPoints(Xsig_aug, delta_t);

    PredictMeanAndCovariance(&x_, &P_);
}

void UKF::UpdateLidar(const MeasurementPackage &m) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
    VectorXd z_pred(2);
    MatrixXd S(2, 2);
    MatrixXd Zsig(2, 2 * n_aug_ + 1);

    PredictLaserMeasurement(&z_pred, &S, &Zsig);
    UpdateState(m, Zsig, z_pred, S, 2);
    CalculateNIS(m, z_pred, S);
}

void UKF::UpdateRadar(const MeasurementPackage &m) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
    VectorXd z_pred(3);
    MatrixXd S(3, 3);
    MatrixXd Zsig(3, 2 * n_aug_ + 1);

    PredictRadarMeasurement(&z_pred, &S, &Zsig);
    UpdateState(m, Zsig, z_pred, S, 3);
    CalculateNIS(m, z_pred, S);
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {
    VectorXd x_aug(7);
    x_aug << x_, 0, 0;

    MatrixXd P_aug = MatrixXd::Zero(7, 7);
    P_aug.block<5, 5>(0, 0) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    MatrixXd A = P_aug.llt().matrixL();

    double op = sqrt(lambda_ + n_aug_);

    MatrixXd c2 = (op * A).colwise() + x_aug;
    MatrixXd c3 = (-op * A).colwise() + x_aug;

    MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug.fill(0.0);

    Xsig_aug.col(0) = x_aug;
    Xsig_aug.block<7, 7>(0, 1) = c2;
    Xsig_aug.block<7, 7>(0, 8) = c3;

    *Xsig_out = Xsig_aug;
}

MatrixXd UKF::PredictSigmaPoints(MatrixXd Xsig_aug, double dt) {
    MatrixXd Xsig_pred(n_x_, n_aug_ * 2 + 1);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        VectorXd x_k = Xsig_aug.col(i);
        double px = x_k(0);
        double py = x_k(0);
        double v = x_k(0);
        double yaw = x_k(0);
        double yawd = x_k(0);

        double px_pred, py_pred = 0;
        if (fabs(yawd) > 0.0001) {
            px_pred = px + (v / yawd) * (sin(yaw + yawd * dt) - sin(yaw));
            px_pred = py + (v / yawd) * (-cos(yaw + yawd * dt) + cos(yaw));
        } else {
            px_pred = px + v * cos(yaw) * dt;
            py_pred = py + v * sin(yaw) * dt;
        }
        VectorXd m(5);
        m << px_pred, py_pred, 0, yawd * dt, 0;

        VectorXd noise(5);
        double t2 = 0.5 * dt * dt;
        noise << t2 * cos(yaw) * std_a_,
                t2 * sin(yaw) * std_a_,
                dt * std_a_,
                t2 * std_yawdd_,
                dt * std_yawdd_;

        Xsig_pred.col(i) = x_k + m + noise;
    }
    return Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd *x_out, MatrixXd *P_out) {

    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }

    VectorXd x = VectorXd::Zero(n_x_);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x += weights_(i) * Xsig_pred_.col(i);
    }
    MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        MatrixXd x_diff = Xsig_pred_.col(i) - x;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        P += weights_(i) * x_diff * x_diff.transpose();
    }

    *x_out = x;
    *P_out = P;
}

void UKF::PredictRadarMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd *Zsig_out) {
    MatrixXd Zsig(3, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd::Zero(3);
    MatrixXd S = MatrixXd::Zero(3, 3);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {

        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double rho = sqrt(px * px + py * py);
        double psi = atan(py / px);
        double rhodot = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;

        VectorXd xz_pred(3);
        xz_pred << rho, psi, rhodot;
        Zsig.col(i) = xz_pred;
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        MatrixXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
        S += weights_(i) * z_diff * z_diff.transpose();
    }
    MatrixXd R = MatrixXd::Zero(3, 3);
    R(0, 0) = std_radr_ * std_radr_;
    R(1, 1) = std_radphi_ * std_radphi_;
    R(2, 2) = std_radrd_ * std_radrd_;

    S += R;

    *z_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}

void UKF::PredictLaserMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd *Zsig_out) {
    MatrixXd Zsig(2, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd::Zero(2);
    MatrixXd S = MatrixXd::Zero(2, 2);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);

        VectorXd xz_pred(2);
        xz_pred << px, py;

        Zsig.col(i) = weights_(i) * xz_pred;
    }
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        MatrixXd z_diff = Zsig.col(i) - z_pred;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd::Zero(2, 2);
    R(0, 0) = std_laspx_;
    R(1, 1) = std_laspy_;

    S += R;

    *z_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}

void UKF::UpdateState(const MeasurementPackage &m, MatrixXd Zsig, const VectorXd &z_pred, const MatrixXd &S, int n_z) {
    MatrixXd Tc(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {

        MatrixXd x_diff = Xsig_pred_.col(i) - x_;
        MatrixXd z_diff = Zsig.col(i) - z_pred;

        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    MatrixXd K = Tc * S.inverse();
    VectorXd z_diff = m.raw_measurements_ - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

}

void UKF::CalculateNIS(MeasurementPackage m, VectorXd z_pred, MatrixXd S) {
    MatrixXd z_diff = z_pred - m.raw_measurements_;
    double eps = (z_diff.transpose() * S.inverse() * z_diff)(0, 0);

    if (m.sensor_type_ == MeasurementPackage::SensorType::LASER) {
        l_eps.push_back(eps);
    } else {
        r_eps.push_back(eps);
    }
}
