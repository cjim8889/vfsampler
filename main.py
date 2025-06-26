import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import optax
import typer
from jaxtyping import PRNGKeyArray

import wandb
from pkg.distributions.annealing import AnnealedDistribution
from pkg.distributions.gaussian import MultivariateGaussian, ZeroMeanMultivariateGaussian
from pkg.distributions.gmm import GMM
from pkg.distributions.multi_double_well import MultiDoubleWellEnergy
from pkg.mcmc.smc import generate_samples_with_smc
from pkg.nn.mlp import MLPVelocityField
from pkg.nn.transformer import ParticleTransformerV4
from pkg.ode.integration import generate_samples
from pkg.training.dt_logZt import estimate_dt_logZt
from pkg.training.objective import Particle, loss_fn
from pkg.training.augmentation import batch_augment_particle_coordinates

app = typer.Typer()

@app.command()
def main(
    batch_size_multiplier: int = typer.Option(16, "--batch-size-multiplier", "-b", help="Multiplier for batch size (batch_size = multiplier * time_steps)"),
    initial_sigma: float = typer.Option(20.0, "--initial-sigma", help="Sigma for initial distribution"),
    dataset_size: int = typer.Option(2560, "--dataset-size", "-d", help="Size of the training dataset"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "-lr", help="Learning rate for training"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay", "-wd", help="Weight decay for training"),
    num_epochs: int = typer.Option(1000, "--epochs", "-e", help="Number of training epochs"),
    time_steps: int = typer.Option(128, "--time-steps", "-t", help="Number of time steps"),
    data_refresh_interval: int = typer.Option(10, "--refresh-interval", "-r", help="Epochs between dataset refreshes"),
    random_seed: int = typer.Option(555, "--seed", help="Random seed for reproducibility"),
    hidden_dim: int = typer.Option(128, "--hidden-dim", "-hd", help="Hidden dimension for the velocity field"),
    depth: int = typer.Option(4, "--depth", "-dp", help="Depth for the velocity field"),
    num_frequencies: int = typer.Option(4, "--num-frequencies", "-nf", help="Number of frequencies for the velocity field"),
    # Data augmentation flags
    enable_rotation: bool = typer.Option(False, "--enable-rotation", help="Enable random rotation augmentation"),
    enable_translation: bool = typer.Option(False, "--enable-translation", help="Enable random translation augmentation"),
    enable_noise: bool = typer.Option(False, "--enable-noise", help="Enable random normal noise augmentation"),
    remove_mean: bool = typer.Option(False, "--remove-mean", help="Remove mean from the training data"),
    translation_scale: float = typer.Option(1.0, "--translation-scale", help="Scale for random translation augmentation"),
    noise_scale: float = typer.Option(1.0, "--noise-scale", help="Scale for random normal noise augmentation"),
):
    """Train an augmented flow sampling model with configurable parameters."""
    
    key = jax.random.PRNGKey(random_seed)
    batch_size = batch_size_multiplier * time_steps
    ts = jnp.linspace(0, 1, time_steps)
    dim = 8

    # initial_distribution = MultivariateGaussian(
    #     sigma=initial_sigma,
    #     mean=0,
    #     dim=dim,
    # )

    initial_distribution = ZeroMeanMultivariateGaussian(
        n_particles=4,
        space_dim=2,
        sigma=initial_sigma,
    )
    target_distribution = MultiDoubleWellEnergy(
        dim=dim,
        n_particles=4,
    )

    annealed_distribution = AnnealedDistribution(
        initial_density=initial_distribution,
        target_density=target_distribution,
    )

    key, subkey = jax.random.split(key)
    # v_theta = MLPVelocityField(
    #     key=subkey,
    #     in_dim=dim,
    #     out_dim=dim,
    #     hidden_dim=hidden_dim,
    #     depth=depth,
    #     dt=0.01,
    # )
    v_theta = ParticleTransformerV4(
        n_spatial_dim=2,
        hidden_size=hidden_dim,
        num_layers=depth,
        num_heads=4,
        key=subkey,
        mp_policy=jmp.Policy(
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            output_dtype=jnp.float32,
        )
    )

    # initial x
    key, subkey = jax.random.split(key)
    initial_x = initial_distribution.sample(subkey, (4000,))

    key, subkey = jax.random.split(key)
    smc_samples = generate_samples_with_smc(
        key=subkey,
        initial_samples=initial_x,
        time_dependent_log_density=annealed_distribution.time_dependent_unnormalised_log_prob,
        ts=ts,
        num_hmc_steps=5,
        num_integration_steps=6,
        step_size=0.1,
    )

    xs = smc_samples["positions"]
    weights = smc_samples["weights"]
    ess = smc_samples["ess"]

    # Visualise the samples
    fig = target_distribution.visualise(xs[-1, :])      
    plt.savefig("smc_samples.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    dt_logZt = estimate_dt_logZt(
        xs=xs,
        weights=weights,
        ts=ts,
        time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
    )

    particles = Particle(
        x=xs.reshape(-1, dim),
        t=jnp.repeat(ts, xs.shape[1]),
        dt_logZt=jnp.repeat(dt_logZt, xs.shape[1]),
    )

    loss, raw_epsilons = loss_fn(
        v_theta=v_theta,
        particles=particles,
        time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
        score_fn=annealed_distribution.score_fn,
        num_frequencies=num_frequencies,
    )

    # Display initial diagnostics
    print(raw_epsilons)
    print(f"Initial loss: {loss}")

    # ================== WANDB INITIALIZATION ==================

    # Initialize wandb for experiment tracking
    # This logs: hyperparameters, training metrics (loss, time), visualizations,
    # dataset refreshes, and final results. Change project name as needed.
    wandb.init( 
        project="variational-flow-sampling",
        config={
            "batch_size": batch_size,
            "batch_size_multiplier": batch_size_multiplier,
            "time_steps": len(ts),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "dataset_size": dataset_size,
            "training_batch_size": batch_size,  # Updated to use batch_size
            "data_refresh_interval": data_refresh_interval,
            "initial_distribution_sigma": initial_sigma,
            "target_distribution_type": "GMM",
            "v_theta_hidden_dim": hidden_dim,
            "v_theta_depth": depth,
            "smc_num_hmc_steps": 5,
            "smc_num_integration_steps": 4,
            "smc_step_size": 0.1,
            "random_seed": random_seed,
            # Data augmentation parameters
            "enable_rotation": enable_rotation,
            "enable_translation": enable_translation,
            "enable_noise": enable_noise,
            "remove_mean": remove_mean,
            "translation_scale": translation_scale,
            "noise_scale": noise_scale,
        }
    )

    # Log initial metrics
    wandb.log({
        "initial_loss": loss,
        "initial_raw_epsilons": raw_epsilons.mean(),
        "initial_ess_mean": ess.mean(),
        "initial_ess_min": ess.min(),
        "initial_ess_final": ess[-1],  # ESS at final time step
    })

    # Log initial SMC visualization
    wandb.log({"initial_smc_samples": wandb.Image("smc_samples.png")})

    # ================== TRAINING BOILERPLATE ==================

    def generate_training_data(
        key: PRNGKeyArray, 
        batch_size: int, 
        initial_distribution: any, 
        annealed_distribution: AnnealedDistribution, 
        ts: jnp.ndarray
    ):
        """Generate full training dataset using SMC sampling."""
        key_initial, key_smc = jax.random.split(key, 2)
        
        # Generate initial samples
        initial_x = initial_distribution.sample(key_initial, (batch_size,))
        
        # Run SMC sampling
        smc_samples = generate_samples_with_smc(
            key=key_smc,
            initial_samples=initial_x,
            time_dependent_log_density=annealed_distribution.time_dependent_unnormalised_log_prob,
            ts=ts,
            num_hmc_steps=5,
            num_integration_steps=4,
            step_size=0.1,
        )
        
        xs = smc_samples["positions"]
        weights = smc_samples["weights"]
        
        dt_logZt = estimate_dt_logZt(
            xs=xs,
            weights=weights,
            ts=ts,
            time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
        )
        
        # Return all particles (flattened across time and trajectory dimensions)
        time_steps, num_particles, dim = xs.shape
        all_particles_flat = xs.reshape(-1, dim)  # (time_steps * num_particles, dim)
        
        particles = Particle(
            x=all_particles_flat,
            t=jnp.repeat(ts, num_particles),
            dt_logZt=jnp.repeat(dt_logZt, num_particles),
        )
        
        return particles

    def sample_batch_from_data(
        key, 
        particles, 
        batch_size,
        enable_rotation: bool = False,
        enable_translation: bool = False,
        enable_noise: bool = False,
        remove_mean: bool = False,
        translation_scale: float = 1.0,
        noise_scale: float = 0.1
    ):
        """Sample a batch from the full training dataset with optional data augmentation."""
        key_sample, key_aug = jax.random.split(key, 2)
        
        total_size = particles.x.shape[0]
        batch_indices = jax.random.choice(
            key_sample, 
            total_size, 
            shape=(min(batch_size, total_size),), 
            replace=False
        )
        
        batch_x = particles.x[batch_indices]
        batch_t = particles.t[batch_indices]
        batch_dt_logZt = particles.dt_logZt[batch_indices]
        
        # Apply data augmentation if any augmentation is enabled
        if enable_rotation or enable_translation or enable_noise:
            # Generate keys for each particle in the batch
            aug_keys = jax.random.split(key_aug, batch_x.shape[0])
            
            # Apply batch augmentation (assuming 4 particles in 2D space)
            batch_x = batch_augment_particle_coordinates(
                batch_x,
                aug_keys,
                4,  # num_particles from target_distribution.n_particles
                2,  # spatial_dim from transformer n_spatial_dim
                enable_rotation,
                enable_translation,
                enable_noise,
                translation_scale,
                noise_scale,
            )

        if remove_mean:
            batch_x = batch_x.reshape(-1, 4, 2)
            batch_x = batch_x - batch_x.mean(axis=1, keepdims=True)
            batch_x = batch_x.reshape(-1, 8)
        
        batch_particles = Particle(
            x=batch_x,
            t=batch_t,
            dt_logZt=batch_dt_logZt,
        )
        
        return batch_particles

    @eqx.filter_jit
    def training_step(state: "TrainState", opt_state, optimizer, particles: Particle):
        """Performs a single gradient step updating both the velocity field and
        the learnable components inside the annealed distribution."""

        def loss_wrapper(model: "TrainState"):
            ad = model.annealed_distribution
            vt = model.v_theta

            loss, _ = loss_fn(
                v_theta=vt,
                particles=particles,
                time_derivative_log_density=ad.unnormalised_time_derivative,
                score_fn=ad.score_fn,
                num_frequencies=num_frequencies,
            )
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_wrapper)(state)

        # Use Equinox helpers to apply the gradient updates
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(state, eqx.is_inexact_array),
        )

        state = eqx.apply_updates(state, updates)
        return state, opt_state, loss

    def visualize_generated_samples(state: "TrainState", key, target_distribution, initial_distribution,
                                   ts, num_vis_samples=5000, epoch=0):
        """Generate samples using current model (velocity field + distribution) and
        save a scatter plot of the generated data."""

        # Create a visualization directory if it doesn't exist
        vis_dir = "training_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate samples using the current v_theta model
        # We need to create a sample function for the initial distribution
        def sample_fn(key, shape):
            key, subkey = jax.random.split(key)
            x = initial_distribution.sample(key, shape)
            return x
        
        # Generate samples by integrating the velocity field
        generated_data = generate_samples(
            key=key,
            v_theta=state.v_theta,
            num_samples=num_vis_samples,
            ts=ts,
            sample_fn=sample_fn,
            save_trajectory=False,
        )
        
        # Extract final positions (remove augmented dimensions)
        final_positions = generated_data["positions"]  # Take final time step
        
        # Visualize using the target distribution's visualization method
        fig = target_distribution.visualise(final_positions)
        plt.title(f"Generated Samples - Epoch {epoch}")
        
        # Save the plot locally and log to wandb
        plot_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Log to wandb
        wandb.log({
            "generated_samples": wandb.Image(plot_path),
            "epoch": epoch
        })
        
        plt.close(fig)
        
        print(f"          | Saved visualization to {plot_path} and logged to wandb")

    # Training configuration
    log_interval = 1
    training_batch_size = batch_size

    print("Training Configuration:")
    print(f"  Batch size: {batch_size} (multiplier: {batch_size_multiplier})")
    print(f"  Initial distribution sigma: {initial_sigma}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Time steps: {time_steps}")
    print(f"  Data refresh interval: {data_refresh_interval}")
    print(f"  Random seed: {random_seed}")
    print(f"  Remove mean: {remove_mean}")
    print("Data Augmentation:")
    print(f"  Rotation: {enable_rotation}")
    print(f"  Translation: {enable_translation} (scale: {translation_scale})")
    print(f"  Noise: {enable_noise} (scale: {noise_scale})")

    # ------------------------------------------------------------------
    # 1.  Bundle all learnable components into a single TrainState so that the
    #    optimiser can simultaneously update both the velocity field and the
    #    parameters inside the (learnable) annealed distribution.
    # ------------------------------------------------------------------

    class TrainState(eqx.Module):
        """Container holding *all* learnable parameters.

        Fields that are Equinox modules are automatically traversed by JAX, so
        gradients will flow into them. Static fields are filtered out via
        `eqx.field(static=True)` in their own class definitions (see
        `AugmentedAnnealedDistribution`)."""

        v_theta: MLPVelocityField
        annealed_distribution: AnnealedDistribution

    # Instantiate the train state
    train_state = TrainState(v_theta=v_theta, annealed_distribution=annealed_distribution)

    # Setup optimizer (after creating train_state so we can initialise its params)
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    grad_clip = optax.clip_by_global_norm(1.0)
    zero_nans = optax.zero_nans()
    optimizer = optax.chain(zero_nans, grad_clip, optimizer)
    opt_state = optimizer.init(eqx.filter(train_state, eqx.is_inexact_array))

    # Training loop
    print("Starting training...")
    print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}, Weight decay: {weight_decay}")
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
                data_key,
                dataset_size,
                initial_distribution,
                train_state.annealed_distribution,
                ts,
            )
            if epoch > 0:
                print(f"Epoch {epoch}: Generated fresh training dataset ({full_dataset.x.shape[0]} particles)")
                # Log dataset refresh to wandb
                wandb.log({
                    "dataset_refresh": 1,
                    "dataset_size": full_dataset.x.shape[0],
                    "epoch": epoch
                })

        # ---------------------------------------------
        # NEW: iterate over multiple gradient steps per epoch
        # ---------------------------------------------
        # steps_per_epoch = max(1, full_dataset.xr.shape[0] // training_batch_size)
        steps_per_epoch = 250
        epoch_losses = []  # store losses to compute an average per epoch

        for step in range(steps_per_epoch):
            # Sample a batch from the full dataset for this training step
            key, batch_key = jax.random.split(key)
            batch_particles = sample_batch_from_data(
                batch_key, 
                full_dataset, 
                training_batch_size,
                enable_rotation,
                enable_translation,
                enable_noise,
                remove_mean,
                translation_scale,
                noise_scale,
            )

            # Training step
            train_state, opt_state, current_loss = training_step(
                train_state, opt_state, optimizer,
                batch_particles
            )

            if step % 10 == 0:
                print(f"Step {step} | Loss: {current_loss:.6f}")
                # Log step-level metrics to wandb
                wandb.log({
                    "step_loss": current_loss,
                    "global_step": epoch * steps_per_epoch + step,
                    "epoch": epoch
                })

            epoch_losses.append(current_loss)

        # Compute average loss over the epoch for logging/early-stopping etc.
        current_loss = jnp.mean(jnp.stack(epoch_losses))

        # Logging
        if epoch % log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch:4d} | Avg Loss: {current_loss:.6f} | Time: {elapsed_time:.1f}s | Steps: {steps_per_epoch}")
            
            # Log epoch-level metrics to wandb
            wandb.log({
                "epoch": epoch,
                "avg_loss": current_loss,
                "elapsed_time": elapsed_time,
                "steps_per_epoch": steps_per_epoch,
            })
            
            if current_loss < best_loss:
                best_loss = current_loss
                print(f"          | New best loss: {best_loss:.6f}")
                # Log new best loss
                wandb.log({
                    "best_loss": best_loss,
                    "epoch": epoch
                })

        # Visualize generated samples every 10 epochs
        if epoch % 3 == 0:
            visualize_generated_samples(train_state, key, target_distribution, initial_distribution, 
                                        ts, epoch=epoch)

    # Final evaluation
    key, test_key = jax.random.split(key)
    test_dataset = generate_training_data(
        test_key,
        dataset_size,
        initial_distribution,
        train_state.annealed_distribution,
        ts,
    )

    # Sample a batch for evaluation
    key, eval_batch_key = jax.random.split(key)
    test_particles = sample_batch_from_data(
        eval_batch_key, 
        test_dataset, 
        training_batch_size,
        enable_rotation,
        enable_translation,
        enable_noise,
        remove_mean,
        translation_scale,
        noise_scale,
    )

    final_loss, _ = loss_fn(
        v_theta=train_state.v_theta,
        particles=test_particles,
        time_derivative_log_density=train_state.annealed_distribution.unnormalised_time_derivative,
        score_fn=train_state.annealed_distribution.score_fn,
        num_frequencies=num_frequencies,
    )

    print("-" * 60)
    print("Training completed!")
    print(f"Initial loss: {loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    # Log final results to wandb
    total_training_time = time.time() - start_time
    wandb.log({
        "final_loss": final_loss,
        "best_loss": best_loss,
        "total_training_time": total_training_time,
        "loss_improvement": loss - final_loss,
        "loss_improvement_ratio": (loss - final_loss) / loss if loss != 0 else 0,
    })

    # Generate and log final visualization
    key, final_vis_key = jax.random.split(key)
    visualize_generated_samples(train_state, final_vis_key, target_distribution, initial_distribution, 
                               ts, epoch=num_epochs, num_vis_samples=10000)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    app()



