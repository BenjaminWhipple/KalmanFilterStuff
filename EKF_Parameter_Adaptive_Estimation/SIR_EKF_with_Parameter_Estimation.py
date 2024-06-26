"""
We roughly follow the paper:
Sun X, Jin L, Xiong M (2008) Extended Kalman Filter for Estimation of Parameters in Nonlinear State-Space Models of Biochemical Networks. PLOS ONE 3(11): e3758. https://doi.org/10.1371/journal.pone.0003758
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

beta = 0.05
gamma = 0.01


# Initial state
S0, I0, R0 = 0.99, 0.01, 0.0  # Assuming a small initial infection in a population normalized to 1
x0 = np.array([S0, I0, R0])

dt = 1.0  # One time unit

num_steps = 500

time = np.arange(0, num_steps+1)

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

t_points = np.linspace(0, 500, num_steps+1)

solution = solve_ivp(sir_model, [0, 500], x0, t_eval=t_points,args=(beta,gamma))

# Extended SIR model, including beta and gamma in the state vector
def extended_sir_model(t, y):
    S, I, R, beta, gamma = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I, 0, 0]

# Used in computing the variance of the predictions. See the referenced paper.
def F_k(y):
    S, I, _, beta, gamma = y
    return np.array([[-beta * I, -beta * S, 0, -S * I, 0],
                     [beta * I, beta * S - gamma, 0, S * I, -I],
                     [0, gamma, 0, 0, I],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])

# Initial state now includes guesses for beta and gamma
S0, I0, R0 = 0.99, 0.01, 0.0
beta_guess, gamma_guess = 0.2, 0.2  # Initial guesses for beta and gamma
x0_extended = np.array([S0, I0, R0, beta_guess, gamma_guess])

true_states=np.concatenate([solution.y, np.full((1,501),beta_guess),np.full((1,501),gamma_guess)],axis=0)
print(true_states)

# Noise matrices
P_extended = np.eye(5) * 0.001
P_extended[3, 3], P_extended[4, 4] = 0,0  # Higher uncertainty in beta and gamma estimates
Q_extended = np.eye(5) * 0.01
R_extended = np.eye(3) * 0.01

# Prediction step for the EKF
def predict(x, P, dt):
    # Integrate the extended SIR model over the interval [0, dt]
    sol = solve_ivp(extended_sir_model, [0, dt], x, method='RK45')
    x_pred = sol.y[:, -1]
    
    # Calculate the linearization for the prediction variance at the current state
    J = F_k(x)
    
    # Predict the error covariance matrix
    P_pred = J @ P @ J.T + Q_extended
    return x_pred, P_pred

# Update step for the EKF
def update(x_pred, P_pred, z):
    # Measurement matrix
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]]) # Only X and Y are measured.

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
x_est = np.array([S0, I0, R0, beta_guess, gamma_guess])
P_est = P_extended

x_est = np.array([S0, I0, R0, beta_guess, gamma_guess])

estimated_states = np.zeros((5, num_steps+1))
estimated_states[:,0]=x0_extended

# Main EKF loop
for step in range(num_steps):
    # Simulate measurement (this should come from real data in practice)
    z_sim = np.array([true_states[0, step] + np.random.normal(0, np.sqrt(R_extended[0, 0])),
                      true_states[1, step] + np.random.normal(0, np.sqrt(R_extended[1, 1])),
                      true_states[2, step] + np.random.normal(0, np.sqrt(R_extended[2, 2]))])
    
    # Prediction step
    x_pred, P_pred = predict(x_est, P_est, dt)
    # Update step
    x_est, P_est = update(x_pred, P_pred, z_sim)
    # Store estimates (for visualization or further processing)
    estimated_states[:, step+1] = x_est

# Visualization
fig, axs = plt.subplots(5, 1, figsize=(5, 10),sharex='col')  # 3 rows, 1 column

# Susceptible
axs[0].plot(time, true_states[0, :], 'b-', label='True Susceptible')
axs[0].plot(time, estimated_states[0, :], 'b--', label='Estimated Susceptible')
axs[0].set_title('SIR EKF (Unknown Parameters)')
axs[0].set_ylabel('Proportion')
axs[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
axs[0].legend(loc="upper right")

# Infected
axs[1].plot(time, true_states[1, :], 'r-', label='True Infected')
axs[1].plot(time, estimated_states[1, :], 'r--', label='Estimated Infected')
axs[1].set_ylabel('Proportion')
axs[1].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
axs[1].legend(loc="upper right")

# Recovered
axs[2].plot(time, true_states[2, :], 'g-', label='True Recovered')
axs[2].plot(time, estimated_states[2, :], 'g--', label='Estimated Recovered')
axs[2].set_ylabel('Proportion')
axs[2].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
axs[2].legend(loc="upper right")

# beta
axs[3].plot(time, np.full((len(time),),beta), 'c-', label=r'True $ \beta $')
axs[3].plot(time, estimated_states[3, :], 'c--', label=r'Estimated $ \beta $')
axs[3].set_ylabel('Estimate')
axs[3].legend(loc="upper right")

# gamma
axs[4].plot(time, np.full((len(time),),gamma), 'm-', label=r'True $ \gamma $')
axs[4].plot(time, estimated_states[4, :], 'm--', label=r'Estimated $ \gamma $')
axs[4].set_xlabel('Time')
axs[4].set_ylabel('Estimate')
axs[4].legend(loc="upper right")

plt.tight_layout()  # Adjust layout to not overlap
plt.savefig('Images/SIR_EKF_UnknownParameters.png',dpi=400)
