import numpy as np
import tensorflow as tf

from models.diff_aug import diff_augment

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """
    This function is pure NumPy, so it remains unchanged.
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def continuous_t_beta(t, T):
        b_max = 5.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        return 1 - alpha

    if beta_schedule == "continuous_t":
        betas = continuous_t_beta(np.arange(1, num_diffusion_timesteps+1), num_diffusion_timesteps)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cosine':
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return betas_clipped
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def q_sample(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise_type='gauss', noise_std=1.0):
    # NOTE: Assumes NCHW data format like the original PyTorch code.
    # If using NHWC, the reshape dimensions would be [-1, 1, 1, 1].
    if noise_type == 'gauss':
        noise = tf.random.normal(shape=tf.shape(x_0)) * noise_std
    elif noise_type == 'bernoulli':
        # Create Bernoulli noise in {-1, 1}
        noise_probs = tf.ones_like(x_0) * 0.5
        noise_samples = tf.cast(tf.random.uniform(shape=tf.shape(x_0)) > 0.5, x_0.dtype)
        noise = (noise_samples * 2.0 - 1.0) * noise_std
    else:
        raise NotImplementedError(noise_type)
        
    # Use tf.gather to select the alphas for the given timesteps t.
    alphas_t_sqrt = tf.gather(alphas_bar_sqrt, t)
    one_minus_alphas_bar_t_sqrt = tf.gather(one_minus_alphas_bar_sqrt, t)
    
    # Reshape for broadcasting. Assumes NCHW data format [N, C, H, W].
    alphas_t_sqrt = tf.reshape(alphas_t_sqrt, [-1, 1, 1, 1])
    one_minus_alphas_bar_t_sqrt = tf.reshape(one_minus_alphas_bar_t_sqrt, [-1, 1, 1, 1])
    
    x_t = alphas_t_sqrt * x_0 + one_minus_alphas_bar_t_sqrt * noise
    return x_t


def q_sample_c(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise_type='gauss', noise_std=1.0):
    # NOTE: Assumes NCHW data format.
    batch_size, num_channels, _, _ = x_0.shape
    if noise_type == 'gauss':
        noise = tf.random.normal(shape=tf.shape(x_0)) * noise_std
    elif noise_type == 'bernoulli':
        noise_probs = tf.ones_like(x_0) * 0.5
        noise_samples = tf.cast(tf.random.uniform(shape=tf.shape(x_0)) > 0.5, x_0.dtype)
        noise = (noise_samples * 2.0 - 1.0) * noise_std
    else:
        raise NotImplementedError(noise_type)
    
    alphas_t_sqrt = tf.gather(alphas_bar_sqrt, t)
    one_minus_alphas_bar_t_sqrt = tf.gather(one_minus_alphas_bar_sqrt, t)
    
    # Reshape for per-channel broadcasting.
    alphas_t_sqrt = tf.reshape(alphas_t_sqrt, [batch_size, num_channels, 1, 1])
    one_minus_alphas_bar_t_sqrt = tf.reshape(one_minus_alphas_bar_t_sqrt, [batch_size, num_channels, 1, 1])
    
    x_t = alphas_t_sqrt * x_0 + one_minus_alphas_bar_t_sqrt * noise
    return x_t


class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, x):
        return x


# Keras/TF models are serializable by default. The @persistence.persistent_class
# decorator is from a specific PyTorch framework and is not needed here.
class Diffusion(tf.keras.layers.Layer):
    def __init__(self,
        beta_schedule='linear', beta_start=1e-4, beta_end=2e-2,
        t_min=10, t_max=1000, noise_std=0.05,
        aug='no', ada_maxp=None, ts_dist='priority',
        **kwargs
    ):
        super(Diffusion, self).__init__(**kwargs)
        self.p = 0.0  # Overall multiplier for augmentation probability.
        self.aug_type = aug
        self.ada_maxp = ada_maxp
        self.noise_type = self.base_noise_type = 'gauss'
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.t_min = t_min
        self.t_max = t_max
        self.t_add = int(t_max - t_min)
        self.ts_dist = ts_dist
        self.aug_policy = 'color,translation'

        # Image-space corruptions.
        self.noise_std = float(noise_std)
        self.noise_type = "gauss"
        
        # Instantiate augmentation layers
        # if aug == 'ada':
        #     self.aug = AdaAugment(p=0.0)
        if aug == 'diff':
            # self.aug = diff_augment()
            self.aug = lambda x: diff_augment(x, policy=self.aug_policy, channels_first=True)
        else:
            self.aug = Identity()

        self.update_T()

    def set_diffusion_process(self, t, beta_schedule):
        betas_np = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            num_diffusion_timesteps=t,
        )

        self.betas = tf.convert_to_tensor(betas_np, dtype=tf.float32)
        self.num_timesteps = self.betas.shape[0]

        self.alphas = 1.0 - self.betas
        
        # tf.math.cumprod is the equivalent of torch.cumprod
        alphas_cumprod = tf.concat([tf.constant([1.0]), tf.math.cumprod(self.alphas)], axis=0)
        self.alphas_bar_sqrt = tf.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = tf.sqrt(1.0 - alphas_cumprod)

    def update_T(self):
        if self.aug_type == 'ada':
            _p = min(self.p, self.ada_maxp) if self.ada_maxp is not None else self.p
            # .assign() is the TF equivalent of PyTorch's in-place .copy_()
            self.aug.p.assign(_p)

        t_adjust = round(self.p * self.t_add)
        t = np.clip(int(self.t_min + t_adjust), a_min=self.t_min, a_max=self.t_max)

        # Update beta values according to new T
        self.set_diffusion_process(t, self.beta_schedule)

        # Sampling t
        self.t_epl = np.zeros(64, dtype=np.int32)
        diffusion_ind = 32
        if self.ts_dist == 'priority':
            # Create a non-zero probability distribution
            prob_range = np.arange(1, t + 1)
            prob_t = prob_range / prob_range.sum()
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind, p=prob_t)
        elif self.ts_dist == 'uniform':
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind)
        else:
            # Fallback for empty/invalid distribution
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind)
            
        self.t_epl[:diffusion_ind] = t_diffusion

    def call(self, x_0):
        x_0_aug = self.aug(x_0)
        
        tf.debugging.assert_rank(x_0_aug, 4, message="Input tensor must be 4D.")
        batch_size = tf.shape(x_0_aug)[0]

        max_index = self.t_epl.shape[0]
        random_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=max_index, dtype=tf.int32)
        
        t = tf.gather(self.t_epl, random_indices)
        
        x_t = q_sample(x_0_aug, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise_type=self.noise_type, noise_std=self.noise_std)
        return x_t, tf.reshape(t, [-1, 1])