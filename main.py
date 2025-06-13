import time
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from pkg.distributions.augmented_annealing import AugmentedAnnealedDistribution
from pkg.distributions.gaussian import MultivariateGaussian
from pkg.distributions.gmm import GMM
from pkg.mcmc.smc import generate_samples_with_smc
from pkg.nn import MLPVelocityField
from pkg.training.dt_logZt import estimate_dt_logZt
from pkg.training.objective import Particle, loss_fn
from pkg.ode.integration import generate_samples

key = jax.random.PRNGKey(0)
batch_size = 32
augmented_dim = 2
ts = jnp.linspace(0, 1, 64)

initial_distribution = MultivariateGaussian(
    sigma=20.,
    mean=0,
    dim=2,
)
augmented_distribution = MultivariateGaussian(
    sigma=20.,
    mean=0,
    dim=augmented_dim,
)
target_distribution = GMM(
    key=key,
    dim=2,
)

annealed_distribution = AugmentedAnnealedDistribution(
    initial_density=initial_distribution,
    target_density=target_distribution,
    augmented_density=augmented_distribution,
    augmented_dim=augmented_dim,
)

key, subkey = jax.random.split(key)
v_theta = MLPVelocityField(
    key=subkey,
    in_dim=2 + augmented_dim,
    out_dim=2 + augmented_dim,
    hidden_dim=128,
    depth=4,
    augmented_dim=augmented_dim,
    dt=0.01,
)

# initial x
key, subkey = jax.random.split(key)
initial_x = initial_distribution.sample(subkey, (batch_size,))

key, subkey = jax.random.split(key)
smc_samples = generate_samples_with_smc(
    key=subkey,
    initial_samples=initial_x,
    time_dependent_log_density=annealed_distribution.time_dependent_log_prob_without_augmentation,
    ts=ts,
    num_hmc_steps=5,
    num_integration_steps=4,
    step_size=0.1,
)

xs = smc_samples["positions"]
weights = smc_samples["weights"]
ess = smc_samples["ess"]

rs = augmented_distribution.sample(key, (xs.shape[0], xs.shape[1]))

xrs = jnp.concatenate([xs, rs], axis=2)


dt_logZt = estimate_dt_logZt(
    xs=xrs,
    weights=smc_samples["weights"],
    ts=ts,
    time_derivative_log_density=annealed_distribution.time_dependent_log_prob,
)

particles = Particle(
    xr=xrs.reshape(-1, 2 + augmented_dim),
    t=jnp.repeat(ts, xrs.shape[1]),
    dt_logZt=jnp.repeat(dt_logZt, xrs.shape[1]),
)

loss, _ = loss_fn(
    v_theta=v_theta,
    particles=particles,
    time_derivative_log_density=annealed_distribution.time_dependent_log_prob,
    score_fn=annealed_distribution.score_fn,
    r_dim=augmented_dim,
)

print(f"Initial loss: {loss}")

# ================== TRAINING BOILERPLATE ==================

def generate_training_data(key, batch_size, initial_distribution, augmented_distribution, 
                          annealed_distribution, ts, augmented_dim):
    """Generate full training dataset using SMC sampling."""
    key_initial, key_smc, key_aug = jax.random.split(key, 3)
    
    # Generate initial samples
    initial_x = initial_distribution.sample(key_initial, (batch_size,))
    
    # Run SMC sampling
    smc_samples = generate_samples_with_smc(
        key=key_smc,
        initial_samples=initial_x,
        time_dependent_log_density=annealed_distribution.time_dependent_log_prob_without_augmentation,
        ts=ts,
        num_hmc_steps=5,
        num_integration_steps=4,
        step_size=0.1,
    )
    
    xs = smc_samples["positions"]
    weights = smc_samples["weights"]
    
    # Generate augmented dimensions
    rs = augmented_distribution.sample(key_aug, (xs.shape[0], xs.shape[1]))
    xrs = jnp.concatenate([xs, rs], axis=2)
    
    # Estimate time derivative of log Z
    dt_logZt = estimate_dt_logZt(
        xs=xrs,
        weights=weights,
        ts=ts,
        time_derivative_log_density=annealed_distribution.time_dependent_log_prob,
    )
    
    # Return all particles (flattened across time and trajectory dimensions)
    time_steps, num_particles, dim = xrs.shape
    all_particles_flat = xrs.reshape(-1, dim)  # (time_steps * num_particles, dim)
    
    particles = Particle(
        xr=all_particles_flat,
        t=jnp.repeat(ts, num_particles),
        dt_logZt=jnp.repeat(dt_logZt, num_particles),
    )
    
    return particles

def sample_batch_from_data(key, particles, batch_size):
    """Sample a batch from the full training dataset."""
    total_size = particles.xr.shape[0]
    batch_indices = jax.random.choice(
        key, 
        total_size, 
        shape=(min(batch_size, total_size),), 
        replace=False
    )
    
    batch_particles = Particle(
        xr=particles.xr[batch_indices],
        t=particles.t[batch_indices],
        dt_logZt=particles.dt_logZt[batch_indices],
    )
    
    return batch_particles

@eqx.filter_jit
def training_step(v_theta, opt_state, optimizer, time_derivative_log_density, score_fn, r_dim, particles):
    """Single training step with gradient computation and parameter update."""
    def loss_wrapper(model):
        loss, _ = loss_fn(
            v_theta=model,
            particles=particles,
            time_derivative_log_density=time_derivative_log_density,
            score_fn=score_fn,
            r_dim=r_dim,
        )
        return loss
    
    loss, grads = eqx.filter_value_and_grad(loss_wrapper)(v_theta)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(v_theta, eqx.is_inexact_array))
    updated_v_theta = eqx.apply_updates(v_theta, updates)
    
    return updated_v_theta, opt_state, loss

def visualize_generated_samples(v_theta, key, target_distribution, initial_distribution, 
                               ts, augmented_dim, num_vis_samples=1000, epoch=0):
    """Generate samples using current v_theta and visualize them."""
    
    # Create a visualization directory if it doesn't exist
    vis_dir = "training_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate samples using the current v_theta model
    # We need to create a sample function for the initial distribution
    def sample_fn(key, shape):
        key, subkey = jax.random.split(key)
        x = initial_distribution.sample(key, shape)
        r = augmented_distribution.sample(subkey, shape)
        xr = jnp.concatenate([x, r], axis=1)
        return xr
    
    # Generate samples by integrating the velocity field
    generated_data = generate_samples(
        key=key,
        v_theta=v_theta,
        num_samples=num_vis_samples,
        ts=ts,
        sample_fn=sample_fn,
        save_trajectory=False,
    )
    
    # Extract final positions (remove augmented dimensions)
    final_positions = generated_data["positions"]  # Take final time step
    final_positions_main = final_positions[:, :2]  # Remove augmented dimensions
    
    # Visualize using the target distribution's visualization method
    fig = target_distribution.visualise(final_positions_main)
    plt.title(f"Generated Samples - Epoch {epoch}")
    
    # Save the plot
    plot_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"          | Saved visualization to {plot_path}")

# Training configuration
learning_rate = 1e-4
num_epochs = 1000
data_refresh_interval = 10  # Regenerate data every N epochs
log_interval = 50
warmup_epochs = 50  # Use smaller learning rate for first N epochs
use_gradient_clipping = True
max_grad_norm = 1.0

# Dataset configuration
dataset_size = 2560  # Generate more particles for the dataset
training_batch_size = 128  # Batch size for each training step

# Setup optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))

# Training loop
print("Starting training...")
print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}")
print(f"Dataset size: {dataset_size}, Training batch size: {training_batch_size}")
print(f"Data refresh interval: {data_refresh_interval} epochs")
print("-" * 60)

start_time = time.time()
best_loss = float('inf')
full_dataset = None

for epoch in range(num_epochs):
    # Regenerate training dataset periodically
    if epoch % data_refresh_interval == 0:
        key, data_key = jax.random.split(key)
        full_dataset = generate_training_data(
            data_key, dataset_size, initial_distribution, augmented_distribution,
            annealed_distribution, ts, augmented_dim
        )
        if epoch > 0:
            print(f"Epoch {epoch}: Generated fresh training dataset ({full_dataset.xr.shape[0]} particles)")
    
    # Sample a batch from the full dataset for this training step
    key, batch_key = jax.random.split(key)
    batch_particles = sample_batch_from_data(batch_key, full_dataset, training_batch_size)
    
    # Training step
    v_theta, opt_state, current_loss = training_step(
        v_theta, opt_state, optimizer,
        annealed_distribution.time_dependent_log_prob,
        annealed_distribution.score_fn,
        augmented_dim,
        batch_particles
    )
    
    # Logging
    if epoch % log_interval == 0:
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch:4d} | Loss: {current_loss:.6f} | Time: {elapsed_time:.1f}s")
        
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"          | New best loss: {best_loss:.6f}")

    # Visualize generated samples every 10 epochs
    if epoch % 10 == 0:
        visualize_generated_samples(v_theta, key, target_distribution, initial_distribution, 
                                    ts, augmented_dim, epoch=epoch)

# Final evaluation
key, test_key = jax.random.split(key)
test_dataset = generate_training_data(
    test_key, dataset_size, initial_distribution, augmented_distribution,
    annealed_distribution, ts, augmented_dim
)

# Sample a batch for evaluation
key, eval_batch_key = jax.random.split(key)
test_particles = sample_batch_from_data(eval_batch_key, test_dataset, training_batch_size)

final_loss, _ = loss_fn(
    v_theta=v_theta,
    particles=test_particles,
    time_derivative_log_density=annealed_distribution.time_dependent_log_prob,
    score_fn=annealed_distribution.score_fn,
    r_dim=augmented_dim,
)

print("-" * 60)
print(f"Training completed!")
print(f"Initial loss: {loss:.6f}")
print(f"Final loss: {final_loss:.6f}")
print(f"Best loss: {best_loss:.6f}")
print(f"Total time: {time.time() - start_time:.1f}s")



