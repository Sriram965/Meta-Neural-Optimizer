import tensorflow as tf

# These are the timescales for our momentum and RMS estimates.
# Each beta value corresponds to a different "memory length":
#   0.9   → remembers roughly the last 10 steps
#   0.99  → remembers roughly the last 100 steps
#   0.999 → remembers roughly the last 1000 steps
MOMENTUM_BETAS = (0.9, 0.99, 0.999)
RMS_BETAS = (0.9, 0.99)

# Total number of features per parameter:
# 2 (log|g|, sign(g)) + 3 (momentum) + 2 (rms) + 1 (timestep) = 8
N_FEATURES = 2 + len(MOMENTUM_BETAS) + len(RMS_BETAS) + 1


def init_states(n_params: int):
    
    momentum_states = {b: tf.zeros([n_params]) for b in MOMENTUM_BETAS}
    rms_states      = {b: tf.zeros([n_params]) for b in RMS_BETAS}
    return momentum_states, rms_states


def update_momentum(grad, momentum_states):
   
    return {
        b: b * momentum_states[b] + (1.0 - b) * grad
        for b in MOMENTUM_BETAS
    }


def update_rms(grad, rms_states):
    
    return {
        b: b * rms_states[b] + (1.0 - b) * grad ** 2
        for b in RMS_BETAS
    }


def compute_features(grad, momentum_states, rms_states, step: int, total_steps: int):
    """
    Build the [n_params, N_FEATURES] input tensor for the MLP optimizer.
    
    This is the function that packages everything the optimizer needs to see
    at each step into a clean, fixed-size feature vector per parameter.
    """

    log_abs_g = tf.math.log(tf.abs(grad) + 1e-8) 
    sign_g    = tf.sign(grad)                    


    m_feats = [momentum_states[b] for b in MOMENTUM_BETAS]

    v_feats = [tf.math.log(rms_states[b] + 1e-8) for b in RMS_BETAS]

    t_feat = tf.ones_like(grad) * (step / max(total_steps - 1, 1))

    # Stacking  all the features along a new axis 
    all_feats = [log_abs_g, sign_g] + m_feats + v_feats + [t_feat]
    return tf.stack(all_feats, axis=1)