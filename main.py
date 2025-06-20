import time
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import wandb
import typer
from jaxtyping import PRNGKeyArray

from pkg.distributions.augmented_annealing import AugmentedAnnealedDistribution
from pkg.distributions.gaussian import MultivariateGaussian
from pkg.distributions.gmm import GMM
from pkg.mcmc.smc import generate_samples_with_smc
from pkg.nn.mlp import AugmentedResidualField
from pkg.training.dt_logZt import estimate_dt_logZt
from pkg.training.objective import Particle, loss_fn
from pkg.ode.integration import generate_samples

app = typer.Typer()

@app.command()
def main(
    batch_size_multiplier: int = typer.Option(16, "--batch-size-multiplier", "-b", help="Multiplier for batch size (batch_size = multiplier * time_steps)"),
    soft_constraint: bool = typer.Option(False, "--soft-constraint", "-s", help="Enable soft constraint"),
    initial_sigma: float = typer.Option(20.0, "--initial-sigma", help="Sigma for initial distribution"),
    augmented_sigma: float = typer.Option(1.0, "--augmented-sigma", help="Sigma for augmented distribution"),
    dataset_size: int = typer.Option(2560, "--dataset-size", "-d", help="Size of the training dataset"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "-lr", help="Learning rate for training"),
    num_epochs: int = typer.Option(1000, "--epochs", "-e", help="Number of training epochs"),
    time_steps: int = typer.Option(128, "--time-steps", "-t", help="Number of time steps"),
    data_refresh_interval: int = typer.Option(10, "--refresh-interval", "-r", help="Epochs between dataset refreshes"),
    random_seed: int = typer.Option(555, "--seed", help="Random seed for reproducibility"),
    hidden_dim: int = typer.Option(128, "--hidden-dim", "-hd", help="Hidden dimension for the velocity field"),
    depth: int = typer.Option(4, "--depth", "-d", help="Depth for the velocity field"),
):
    """Train an augmented flow sampling model with configurable parameters."""
    
    key = jax.random.PRNGKey(random_seed)
    batch_size = batch_size_multiplier * time_steps
    augmented_dim = 2
    ts = jnp.linspace(0, 1, time_steps)

    initial_distribution = MultivariateGaussian(
        sigma=initial_sigma,
        mean=0,
        dim=2,
    )
    augmented_distribution = MultivariateGaussian(
        dim=augmented_dim,
        mean=0,
        sigma=augmented_sigma,
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
        is_conditional=False,
    )

    key, subkey = jax.random.split(key)
    v_theta = AugmentedResidualField(
        key=subkey,
        x_dim=2,
        hidden_dim=hidden_dim,
        augmented_dim=augmented_dim,
        depth=depth,
        f_natural=jax.grad(lambda r: -augmented_distribution.log_prob(r)),
        v_natural=lambda x, t: jax.grad(lambda x: annealed_distribution.time_dependent_log_prob_without_augmentation(x, t))(x),
        dt=0.01,
    )

    # initial x
    key, subkey = jax.random.split(key)
    initial_x = initial_distribution.sample(subkey, (4000,))
    initial_r = augmented_distribution.sample(subkey, (4000,))

    key, subkey = jax.random.split(key)
    smc_samples = generate_samples_with_smc(
        key=subkey,
        initial_samples=jnp.concatenate([initial_x, initial_r], axis=1),
        time_dependent_log_density=annealed_distribution.time_dependent_log_prob,
        ts=ts,
        num_hmc_steps=5,
        num_integration_steps=4,
        step_size=0.1,
    )

    xrs = smc_samples["positions"]
    weights = smc_samples["weights"]
    ess = smc_samples["ess"]

    # Visualise the samples
    fig = target_distribution.visualise(xrs[-1, :, :2])
    plt.savefig("smc_samples.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    dt_logZt = estimate_dt_logZt(
        xs=xrs,
        weights=weights,
        ts=ts,
        time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
    )

    particles = Particle(
        xr=xrs.reshape(-1, 2 + augmented_dim),
        t=jnp.repeat(ts, xrs.shape[1]),
        dt_logZt=jnp.repeat(dt_logZt, xrs.shape[1]),
    )

    loss, raw_epsilons = loss_fn(
        v_theta=v_theta,
        particles=particles,
        time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
        score_fn=annealed_distribution.score_fn,
        r_dim=augmented_dim,
    )

    # Display initial diagnostics
    print(raw_epsilons)
    print(f"Initial loss: {loss}")

    # ================== WANDB INITIALIZATION ==================

    # Initialize wandb for experiment tracking
    # This logs: hyperparameters, training metrics (loss, time), visualizations,
    # dataset refreshes, and final results. Change project name as needed.
    wandb.init(
        project="augmented-flow-sampling",
        config={
            "batch_size": batch_size,
            "batch_size_multiplier": batch_size_multiplier,
            "augmented_dim": augmented_dim,
            "time_steps": len(ts),
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "dataset_size": dataset_size,
            "training_batch_size": batch_size,  # Updated to use batch_size
            "data_refresh_interval": data_refresh_interval,
            "initial_distribution_sigma": initial_sigma,
            "augmented_distribution_sigma": augmented_sigma,
            "target_distribution_type": "GMM",
            "v_theta_hidden_dim": hidden_dim,
            "v_theta_depth": depth,
            "smc_num_hmc_steps": 5,
            "smc_num_integration_steps": 4,
            "smc_step_size": 0.1,
            "soft_constraint": soft_constraint,
            "random_seed": random_seed,
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

    def generate_training_data(key: PRNGKeyArray, batch_size: int, initial_distribution: any, augmented_distribution: any, 
                              annealed_distribution: AugmentedAnnealedDistribution, ts: jnp.ndarray, augmented_dim: int):
        """Generate full training dataset using SMC sampling."""
        key_initial, key_initial_r, key_smc, key_aug = jax.random.split(key, 4)
        
        # Generate initial samples
        initial_x = initial_distribution.sample(key_initial, (batch_size,))
        initial_r = augmented_distribution.sample(key_initial_r, (batch_size,))

        iniital_samples = jnp.concatenate([initial_x, initial_r], axis=1)
        
        # Run SMC sampling
        smc_samples = generate_samples_with_smc(
            key=key_smc,
            initial_samples=iniital_samples,
            time_dependent_log_density=annealed_distribution.time_dependent_log_prob,
            ts=ts,
            num_hmc_steps=5,
            num_integration_steps=4,
            step_size=0.1,
        )
        
        xrs = smc_samples["positions"]
        weights = smc_samples["weights"]
        
        dt_logZt = estimate_dt_logZt(
            xs=xrs,
            weights=weights,
            ts=ts,
            time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
        )

        print(dt_logZt)
        
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
    def training_step(state: "TrainState", opt_state, optimizer, r_dim: int, particles: Particle):
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
                r_dim=r_dim,
                soft_constraint=soft_constraint,
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
                                   ts, augmented_dim, num_vis_samples=5000, epoch=0):
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
            r = augmented_distribution.sample(subkey, shape)
            return jnp.concatenate([x, r], axis=1)
        
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
        final_positions_main = final_positions[:, :2]  # Remove augmented dimensions
        
        # Visualize using the target distribution's visualization method
        fig = target_distribution.visualise(final_positions_main)
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
    print(f"  Soft constraint: {soft_constraint}")
    print(f"  Initial distribution sigma: {initial_sigma}")
    print(f"  Augmented distribution sigma: {augmented_sigma}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Time steps: {time_steps}")
    print(f"  Data refresh interval: {data_refresh_interval}")
    print(f"  Random seed: {random_seed}")

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

        v_theta: AugmentedResidualField
        annealed_distribution: AugmentedAnnealedDistribution

    # Instantiate the train state
    train_state = TrainState(v_theta=v_theta, annealed_distribution=annealed_distribution)

    # Setup optimizer (after creating train_state so we can initialise its params)
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(train_state, eqx.is_inexact_array))

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
                data_key,
                dataset_size,
                initial_distribution,
                train_state.annealed_distribution.augmented_density,
                train_state.annealed_distribution,
                ts,
                augmented_dim,
            )
            if epoch > 0:
                print(f"Epoch {epoch}: Generated fresh training dataset ({full_dataset.xr.shape[0]} particles)")
                # Log dataset refresh to wandb
                wandb.log({
                    "dataset_refresh": 1,
                    "dataset_size": full_dataset.xr.shape[0],
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
            batch_particles = sample_batch_from_data(batch_key, full_dataset, training_batch_size)

            # Training step
            train_state, opt_state, current_loss = training_step(
                train_state, opt_state, optimizer,
                augmented_dim,
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
                                        ts, augmented_dim, epoch=epoch)

    # Final evaluation
    key, test_key = jax.random.split(key)
    test_dataset = generate_training_data(
        test_key,
        dataset_size,
        initial_distribution,
        train_state.annealed_distribution.augmented_density,
        train_state.annealed_distribution,
        ts,
        augmented_dim,
    )

    # Sample a batch for evaluation
    key, eval_batch_key = jax.random.split(key)
    test_particles = sample_batch_from_data(eval_batch_key, test_dataset, training_batch_size)

    final_loss, _ = loss_fn(
        v_theta=train_state.v_theta,
        particles=test_particles,
        time_derivative_log_density=train_state.annealed_distribution.unnormalised_time_derivative,
        score_fn=train_state.annealed_distribution.score_fn,
        r_dim=augmented_dim,
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
                               ts, augmented_dim, epoch=num_epochs, num_vis_samples=10000)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    app()



