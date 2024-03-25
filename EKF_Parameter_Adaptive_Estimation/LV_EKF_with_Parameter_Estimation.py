"""
We roughly follow the paper:
Sun X, Jin L, Xiong M (2008) Extended Kalman Filter for Estimation of Parameters in Nonlinear State-Space Models of Biochemical Networks. PLOS ONE 3(11): e3758. https://doi.org/10.1371/journal.pone.0003758
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lotka-Volterra System

x_0, y_0 = 1.0, 0.5  # Example initial conditions

# Parameters
alpha, beta, delta, gamma = 0.2, 0.2, 0.2, 0.1  # Example parameters

x0 = np.array([x_0,y_0])

dt = 1.0  # One time unit

num_steps = 200

time = np.arange(0, num_steps+1)

def lv_model(t, y, alpha, beta, delta, gamma):
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

t_points = np.linspace(0, num_steps, num_steps+1)

solution = solve_ivp(lv_model, [0, num_steps], x0, t_eval=t_points,args=(alpha, beta, delta, gamma))

# Extended LV model, including the parameters in the state vector
def extended_LV_model(t, y):
    X, Y, alpha, beta, delta, gamma = y
    dX = alpha * X - beta * X * Y
    dY = delta * X * Y - gamma * Y
    
    return [dX,dY, 0, 0, 0, 0]

# See the referenced paper.
def F_k(y):
    X, Y, alpha, beta, delta, gamma = y
    return np.array([[alpha-beta*Y, -beta*Y, X, -X*Y, 0, 0],
                     [delta*Y, delta*X-gamma, 0, 0, X*Y, -Y],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

# Initial state now includes guesses for beta and gamma
x_0, y_0 = 0.5, 0.5
alpha_guess, beta_guess, delta_guess, gamma_guess = 0.1, 0.1, 0.1, 0.1  # Initial guesses
x0_extended = np.array([x_0, y_0, alpha_guess, beta_guess, delta_guess, gamma_guess])

true_states=np.concatenate([solution.y,np.full((1,num_steps+1),alpha_guess),np.full((1,num_steps+1),beta_guess),np.full((1,num_steps+1),delta_guess),np.full((1,num_steps+1),gamma_guess)],axis=0)
print(true_states)
# Adjust the covariance matrices to account for the uncertainties in beta and gamma

P_extended = np.eye(6) * 0.001
P_extended[3, 3], P_extended[4, 4],P_extended[5, 5] = 0,0,0
Q_extended = np.eye(6) * 0.001
R_extended = np.eye(2) * 0.001

# Prediction step for the EKF
def predict(x, P, dt):
    # Integrate the extended SIR model over the interval [0, dt]
    sol = solve_ivp(extended_LV_model, [0, dt], x, method='RK45')
    x_pred = sol.y[:, -1]
    
    # Calculate F_k at the current state
    J = F_k(x)
    
    # Predict the error covariance matrix
    P_pred = J @ P @ J.T + Q_extended
    return x_pred, P_pred

# Update step for the EKF
def update(x_pred, P_pred, z):
    # Measurement matrix
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]]) # Only X an Y are measured.
    
    # Calculate the Kalman Gain
    S = H @ P_pred @ H.T + R_extended
    
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update the state estimate
    z_pred = H @ x_pred
    x_upd = x_pred + K @ (z - z_pred)
    
    # Update the error covariance matrix
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_upd, P_upd

# Initialize state and covariance matrices
x_est = np.array([x_0, y_0, alpha_guess, beta_guess, delta_guess, gamma_guess])
P_est = P_extended

estimated_states = np.zeros((6, num_steps+1))
estimated_states[:,0]=x0_extended

# Main EKF loop
for step in range(num_steps):
    # Simulate measurement (this should come from real data in practice)
    z_sim = np.array([true_states[0, step] + np.random.normal(0, np.sqrt(R_extended[0, 0])),
                      true_states[1, step] + np.random.normal(0, np.sqrt(R_extended[1, 1]))])
    
    # Prediction step
    x_pred, P_pred = predict(x_est, P_est, dt)
    
    # Update step
    x_est, P_est = update(x_pred, P_pred, z_sim)
    
    # Store estimates (for visualization or further processing)
    estimated_states[:, step+1] = x_est

# Visualization
fig, axs = plt.subplots(6, 1, figsize=(5, 10),sharex='col')  # 3 rows, 1 column

axs[0].plot(time, true_states[0, :], 'b-', label='True X')
axs[0].plot(time, estimated_states[0, :], 'b--', label='Estimated X')
axs[0].set_title('Lotka-Volterra EKF (Unknown Parameters)')
axs[0].set_ylabel('Population')
axs[0].legend(loc="upper right")

axs[1].plot(time, true_states[1, :], 'r-', label='True Y')
axs[1].plot(time, estimated_states[1, :], 'r--', label='Estimated Y')
axs[1].set_ylabel('Population')
axs[1].legend(loc="upper right")

axs[2].plot(time, np.full((len(time),),alpha), 'c-', label=r'True $ \alpha $')
axs[2].plot(time, estimated_states[3, :], 'c--', label=r'Estimated $ \alpha $')
axs[2].set_ylabel('Proportion')
axs[2].legend(loc="upper right")

axs[3].plot(time, np.full((len(time),),beta), 'c-', label=r'True $ \beta $')
axs[3].plot(time, estimated_states[3, :], 'c--', label=r'Estimated $ \beta $')
axs[3].set_ylabel('Estimate')
axs[3].legend(loc="upper right")

axs[4].plot(time, np.full((len(time),),delta), 'm-', label=r'True $ \delta $')
axs[4].plot(time, estimated_states[4, :], 'm--', label=r'Estimated $ \delta $')
axs[4].set_xlabel('Time')
axs[4].set_ylabel('Estimate')
axs[4].legend(loc="upper right")

axs[5].plot(time, np.full((len(time),),gamma), 'm-', label=r'True $ \gamma $')
axs[5].plot(time, estimated_states[5, :], 'm--', label=r'Estimated $ \gamma $')
axs[5].set_xlabel('Time')
axs[5].set_ylabel('Estimate')
axs[5].legend(loc="upper right")

plt.tight_layout()  # Adjust layout to not overlap
plt.savefig('Images/LV_EKF_UnknownParameters.png',dpi=400)

