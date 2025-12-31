import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def extract(a, t, x_shape):
    """
    This function abstracts away the tedious indexing that would otherwise have
    to be done to properly compute the diffusion equations from lecture. This
    is necessary because we train data in batches, while the math taught in
    lecture only considers a single sample.
    
    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        extract(alpha, t, x.shape) * x

    Args:
        a: 1D tensor containing the value at each time step.
        t: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: Number of timesteps int. 
        s: The strength of the schedule.
    Returns:
        1D tensor of the same shape as timesteps, with the computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
        # Hint: helpful to define both alpha_bar and alpha_bar_t_minus_1
        # Hint: at t = 0, alpha_bar = 1
        alpha_t = cosine_schedule(self.num_timesteps) #(T,)

        # cumulative products alpha_bar
        alpha_bar = torch.cumprod(alpha_t, dim=0) #(T,)
        alpha_bar_t_minus_1 = torch.cat([torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]], dim=0)  # (T,)

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_t_minus_1", alpha_bar_t_minus_1)

        self.register_buffer("sqrt_alpha_bar",
            torch.sqrt(alpha_bar.clamp_min(0.0)))
        self.register_buffer("sqrt_one_minus_alpha_bar",
            torch.sqrt((1.0 - alpha_bar).clamp_min(0.0)))
        self.register_buffer("sqrt_alpha_bar_t_minus_1",
            torch.sqrt(alpha_bar_t_minus_1.clamp_min(0.0)))
        self.register_buffer("sqrt_alpha_t",
            torch.sqrt(alpha_t.clamp_min(0.0)))        

        posterior_variance = ((1.0-alpha_bar_t_minus_1)/(1-alpha_bar))*(1-alpha_t)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20)) #in case nan from sqrt(posterior_variance <= 0) because of tiny division may contribute to very large value and overflow to negative value.


        # ###########################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled image at timestep t_index + 1.
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled image at timestep t_index.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        eps_hat = self.model(x, t)

        # estimate x0_hat
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, t, x.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bar, t, x.shape)
        x0_hat = (x - (sqrt_one_minus_alpha_bar_t * eps_hat)) / sqrt_alpha_bar_t
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        # estimate mean: u_hat
        u_alpha_t = extract(self.alpha_t, t, x.shape)
        u_sqrt_alpha_t = extract(self.sqrt_alpha_t, t, x.shape)
        u_alpha_bar_t = extract(self.alpha_bar, t, x.shape)
        u_alpha_bar_t_minus_1 = extract(self.alpha_bar_t_minus_1, t, x.shape)
        u_sqrt_alpha_bar_t_minus_1 = extract(self.sqrt_alpha_bar_t_minus_1, t, x.shape)
        u_denom = (1.0 - u_alpha_bar_t)
        u_hat = (u_sqrt_alpha_t * (1.0 - u_alpha_bar_t_minus_1) / u_denom *      x) + \
                (u_sqrt_alpha_bar_t_minus_1 * (1.0 - u_alpha_t) / u_denom * x0_hat)

        # estimate variance: var
        var = extract(self.posterior_variance, t, x.shape)

        if t_index == 0:
            return u_hat
        else:
            return u_hat + torch.sqrt(var) * self.noise_like(x.shape, x.device)
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            img: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generated image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        img = torch.clamp(img, -1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Wrapper function for p_sample_loop.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the sample function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        device = next(self.model.parameters()).device
        noise = self.noise_like((batch_size, self.channels, self.image_size, self.image_size), device)
        return self.p_sample_loop(noise)

    # forward diffusion
    def q_sample(self, x_0, t, noise):
        """
        Applies alpha interpolation between x_0 and noise to simulate sampling
        x_t from the noise distribution.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled images.
        """
        ###### TODO: Implement the q_sample function #######
        x_t = extract(self.sqrt_alpha_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape) * noise
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        x_t = self.q_sample(x_0, t, noise)
        eps_hat = self.model(x_t, t)
        loss = F.l1_loss(eps_hat, noise)
        return loss
        #####################################################

    def forward(self, x_0, noise):
        """
        Acts as a wrapper for p_losses.
        Args:
            x_0: The initial images. (B, C, H, W)
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device, dtype=torch.long)
        return self.p_losses(x_0, t, noise)
