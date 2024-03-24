import numpy as np
from scipy.integrate import solve_ivp  # For solving differential equations
import matplotlib.pyplot as plt

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def jacobian_sir(y, beta, gamma):
    S, I, _ = y
    J = np.array([[-beta * I, -beta * S, 0],
                  [beta * I, beta * S - gamma, 0],
                  [0, gamma, 0]])
    return J

# SIR model parameters
beta = 0.03
gamma = 0.01

# Initial state
S0, I0, R0 = 0.99, 0.01, 0.0  # Assuming a small initial infection in a population normalized to 1
x0 = np.array([S0, I0, R0])

# Time step
dt = 1.0  # One time unit

# Error covariance matrices
P = np.eye(3) * 0.1  # Initial estimate error covariance
Q = np.eye(3) * 0.01  # Process noise covariance (model uncertainty)
R = np.eye(2) * 0.02  # Measurement noise covariance (only for I and R measurements)

def predict(x, P, beta, gamma, dt):
    # Predict state using the SIR model
    t_span = [0, dt]
    sol = solve_ivp(sir_model, t_span, x, args=(beta, gamma), method='RK45')
    x_pred = sol.y[:, -1]
    
    # Predict error covariance
    J = jacobian_sir(x, beta, gamma)
    P_pred = J @ P @ J.T + Q
    
    return x_pred, P_pred

def update(x_pred, P_pred, z, R):
    H = np.array([[0, 1, 0],  # Measurement model: we only measure I and R
                  [0, 0, 1]])
    
    # Kalman gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update state
    z_pred = H @ x_pred
    x_upd = x_pred + K @ (z - z_pred)
    
    # Update error covariance
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    
    return x_upd, P_upd

# Simulate some synthetic measurements for infected and recovered
num_steps = 500
true_states = np.empty((3, num_steps))
measurements = np.empty((2, num_steps))
x = x0
for i in range(num_steps):
    x, _ = predict(x, P, beta, gamma, dt)
    true_states[:, i] = x
    measurements[:, i] = x[1:] + np.random.normal(0, np.sqrt(R.diagonal()), 2)  # Add noise to I and R

# Apply EKF
estimated_states = np.empty((3, num_steps))
x_est = x0
P_est = P
for i in range(num_steps):
    x_pred, P_pred = predict(x_est, P_est, beta, gamma, dt)
    x_est, P_est = update(x_pred, P_pred, measurements[:, i], R)
    estimated_states[:, i] = x_est

time = np.arange(0, num_steps)

fig, axs = plt.subplots(3, 1, figsize=(5, 10))  # 3 rows, 1 column

# Susceptible
axs[0].plot(time, true_states[0, :], 'b-', label='True Susceptible')
axs[0].plot(time, estimated_states[0, :], 'b--', label='Estimated Susceptible')
axs[0].set_title('SIR Extended Kalman Filter Example (Known Parameters)')
axs[0].set_ylabel('Fraction')
axs[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
axs[0].legend()
axs[0].grid(True)

# Infected
axs[1].plot(time, true_states[1, :], 'r-', label='True Infected')
axs[1].plot(time, estimated_states[1, :], 'r--', label='Estimated Infected')
#axs[1].set_title('Infected Population')
axs[1].set_ylabel('Fraction')
axs[1].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
axs[1].legend()
axs[1].grid(True)

# Recovered
axs[2].plot(time, true_states[2, :], 'g-', label='True Recovered')
axs[2].plot(time, estimated_states[2, :], 'g--', label='Estimated Recovered')
#axs[2].set_title('Recovered Population')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Fraction')
axs[2].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()  # Adjust layout to not overlap
plt.show()

"""
## Now, try to estimate beta and gamma simultaniously

def extended_sir_model(t, y):
    # Extended state includes S, I, R, beta, and gamma
    S, I, R, beta, gamma = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    dbetadt = 0  # Beta is assumed constant, so its derivative is zero
    dgammadt = 0  # Gamma is assumed constant, so its derivative is zero
    return [dSdt, dIdt, dRdt, dbetadt, dgammadt]

def jacobian_extended(y):
    S, I, _, beta, gamma = y
    J = np.zeros((5, 5))
    J[0, :] = [-beta * I, -beta * S, 0, -S * I, 0]
    J[1, :] = [beta * I, beta * S - gamma, 0, S * I, -I]
    J[2, :] = [0, gamma, 0, 0, I]
    J[3, :] = [0, 0, 0, 0, 0]  # Derivative of beta with respect to everything is zero
    J[4, :] = [0, 0, 0, 0, 0]  # Derivative of gamma with respect to everything is zero
    return J


# Initial state now includes guesses for beta and gamma
S0, I0, R0 = 0.99, 0.01, 0.0
beta_guess, gamma_guess = 0.25, 0.05  # Initial guesses for beta and gamma
x0_extended = np.array([S0, I0, R0, beta_guess, gamma_guess])

# Adjust the covariance matrices to account for the uncertainties in beta and gamma
P_extended = np.eye(5) * 0.1
P_extended[3, 3], P_extended[4, 4] = 0.1, 0.1  # Higher uncertainty in beta and gamma estimates
Q_extended = np.eye(5) * 0.01
R_extended = np.eye(2) * 0.02  # Measurement noise stays the same

def predict_ext(x, P, dt):
    # Predict state using the SIR model
    t_span = [0, dt]
    sol = solve_ivp(extended_sir_model, t_span, x, method='RK45')
    x_pred = sol.y[:, -1]
    
    # Predict error covariance
    J = jacobian_extended(x)
    P_pred = J @ P_extended @ J.T + Q_extended
    
    return x_pred, P_pred

def update_ext(x_pred, P_pred, z, R):
    H = np.array([[0, 1, 0],  # Measurement model: we only measure I and R
                  [0, 0, 1]])
    
    # Kalman gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update state
    z_pred = H @ x_pred
    x_upd = x_pred + K @ (z - z_pred)
    
    # Update error covariance
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    
    return x_upd, P_upd
"""
