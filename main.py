import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jmp
import matplotlib.pyplot as plt
import optax
import typer
from jaxtyping import PRNGKeyArray

import wandb
from pkg.distributions.annealing import AnnealedDistribution
from pkg.distributions.gaussian import MultivariateGaussian
from pkg.distributions.gmm import GMM
from pkg.distributions.multi_double_well import MultiDoubleWellEnergy
from pkg.mcmc.smc import generate_samples_with_smc
from pkg.nn.mlp import MLPVelocityField
from pkg.nn.test_function import TrainableTestFunction, FixedTestFunction
from pkg.nn.transformer import ParticleTransformerV4
from pkg.ode.integration import generate_samples
from pkg.training.augmentation import batch_augment_particle_coordinates
from pkg.training.dt_logZt import estimate_dt_logZt
from pkg.training.objective import Particle, loss_fn

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
    steps_per_epoch: int = typer.Option(250, "--steps-per-epoch", "-spe", help="Number of steps per epoch"),
    num_particles: int = typer.Option(4, "--num-particles", "-np", help="Number of particles in the target distribution"),
    alternating_frequency: int = typer.Option(50, "--alternating-frequency", "-af", help="Number of steps to train one component before switching"),
    test_fn_learning_rate: float = typer.Option(1e-3, "--test-fn-lr", help="Learning rate for test function (adversarial)"),
    test_fn_l2_reg: float = typer.Option(1e-3, "--test-fn-l2", help="L2 regularization for test function"),
    train_test_fn: bool = typer.Option(True, "--train-test-fn", help="Whether to train the test function (if it has trainable parameters)"),
):
    """Train an augmented flow sampling model with configurable parameters."""
    
    key = jax.random.PRNGKey(random_seed)
    batch_size = batch_size_multiplier * time_steps
    ts = jnp.linspace(0, 1, time_steps)
    dim = num_particles * 2

    initial_distribution = MultivariateGaussian(
        sigma=initial_sigma,
        mean=0,
        dim=dim,
    )

    # initial_distribution = ZeroMeanMultivariateGaussian(
    #     n_particles=4,
    #     space_dim=2,
    #     sigma=initial_sigma,
    # )
    target_distribution = MultiDoubleWellEnergy(
        dim=dim,
        n_particles=num_particles,
    )
    # target_distribution = GMM(
    #     key=key,
    #     dim=dim,
    # )

    annealed_distribution = AnnealedDistribution(
        initial_density=initial_distribution,
        target_density=target_distribution,
    )

    # Create trainable test function
    key, test_fn_key = jax.random.split(key)
    # test_fn = TrainableTestFunction(
    #     key=test_fn_key,
    #     in_dim=dim,
    #     hidden_dim=hidden_dim // 2,  # Use smaller network for test function
    #     depth=2,
    # )

    # Alternative: Use a fixed (non-trainable) test function
    test_fn = FixedTestFunction(
        log_prob_fn=annealed_distribution.time_dependent_unnormalised_log_prob,
        temperatures=[1.0, 5.0, 10.0, 20.0, 100.0, 0.1,],
    )
    # When using FixedTestFunction, the training loop will automatically
    # detect that it has no trainable parameters and skip test function optimization

    key, subkey = jax.random.split(key)

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

    loss, raw_epsilons, phi_values = loss_fn(
        v_theta=v_theta,
        particles=particles,
        time_derivative_log_density=annealed_distribution.unnormalised_time_derivative,
        score_fn=annealed_distribution.score_fn,
        test_fn=test_fn,
    )

    # Display initial diagnostics
    print(raw_epsilons)
    print(phi_values)
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
            "num_particles": num_particles,
            "alternating_frequency": alternating_frequency,
            "test_fn_learning_rate": test_fn_learning_rate,
            "test_fn_l2_reg": test_fn_l2_reg,
            "train_test_fn": train_test_fn,
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

    def is_test_fn_trainable(test_fn):
        """Check if the test function has any trainable parameters."""
        trainable_params = eqx.filter(test_fn, eqx.is_inexact_array)
        return jtu.tree_leaves(trainable_params) != []

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
    def training_step_v_theta(state: "TrainState", v_theta_opt_state, v_theta_optimizer, particles: Particle):
        """Training step for velocity field (minimize loss)."""

        def loss_wrapper_v_theta(v_theta):
            # Create new state with updated v_theta
            new_state = eqx.tree_at(lambda s: s.v_theta, state, v_theta)
            ad = new_state.annealed_distribution
            tf = new_state.test_fn

            loss, _, _ = loss_fn(
                v_theta=v_theta,
                particles=particles,
                time_derivative_log_density=ad.unnormalised_time_derivative,
                score_fn=ad.score_fn,
                test_fn=tf,
            )
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_wrapper_v_theta)(state.v_theta)

        # Update only v_theta
        updates, v_theta_opt_state = v_theta_optimizer.update(
            grads,
            v_theta_opt_state,
            eqx.filter(state.v_theta, eqx.is_inexact_array),
        )

        new_v_theta = eqx.apply_updates(state.v_theta, updates)
        new_state = eqx.tree_at(lambda s: s.v_theta, state, new_v_theta)
        
        return new_state, v_theta_opt_state, loss

    @eqx.filter_jit  
    def training_step_test_fn(state: "TrainState", test_fn_opt_state, test_fn_optimizer, particles: Particle, l2_reg: float):
        """Training step for test function (maximize loss + L2 regularization)."""

        def loss_wrapper_test_fn(test_fn):
            # Create new state with updated test_fn
            new_state = eqx.tree_at(lambda s: s.test_fn, state, test_fn)
            ad = new_state.annealed_distribution
            vt = new_state.v_theta

            # Compute main loss and get phi values for regularization
            main_loss, raw_epsilons, phi_values = loss_fn(
                v_theta=vt,
                particles=particles,
                time_derivative_log_density=ad.unnormalised_time_derivative,
                score_fn=ad.score_fn,
                test_fn=test_fn,
            )
            
            # L2 regularization on test function outputs (phi values)
            l2_penalty = jnp.mean(jnp.square(phi_values))
            
            # Adversarial objective: maximize main loss, minimize L2 penalty on outputs
            adversarial_loss = -main_loss + l2_reg * l2_penalty
            return adversarial_loss, (main_loss, l2_penalty)

        (adversarial_loss, (main_loss, l2_penalty)), grads = eqx.filter_value_and_grad(loss_wrapper_test_fn, has_aux=True)(state.test_fn)

        # Update only test_fn
        updates, test_fn_opt_state = test_fn_optimizer.update(
            grads,
            test_fn_opt_state,
            eqx.filter(state.test_fn, eqx.is_inexact_array),
        )

        new_test_fn = eqx.apply_updates(state.test_fn, updates)
        new_state = eqx.tree_at(lambda s: s.test_fn, state, new_test_fn)
        
        return new_state, test_fn_opt_state, adversarial_loss, main_loss, l2_penalty

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

    # Check if test function training is possible and enabled
    test_fn_is_trainable = is_test_fn_trainable(test_fn)
    test_fn_training_enabled = train_test_fn and test_fn_is_trainable

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
    print(f"  Number of particles: {num_particles}")
    print(f"  Alternating frequency: {alternating_frequency}")
    print(f"  Test function LR: {test_fn_learning_rate}")
    print(f"  Test function L2 reg: {test_fn_l2_reg}")
    print(f"  Test function trainable: {test_fn_is_trainable}")
    print(f"  Test function training enabled: {test_fn_training_enabled}")
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
        test_fn: TrainableTestFunction

    # Instantiate the train state
    train_state = TrainState(v_theta=v_theta, annealed_distribution=annealed_distribution, test_fn=test_fn)

    # Setup two separate optimizers
    # Optimizer for velocity field and annealed distribution (minimize loss)
    v_theta_optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    v_theta_grad_clip = optax.clip_by_global_norm(1.0)
    v_theta_zero_nans = optax.zero_nans()
    v_theta_optimizer = optax.chain(v_theta_zero_nans, v_theta_grad_clip, v_theta_optimizer)
    
    # Optimizer for test function (adversarial: maximize loss, minimize L2)
    # Only create if test function training is enabled
    if test_fn_training_enabled:
        test_fn_optimizer = optax.adamw(test_fn_learning_rate, weight_decay=0.0)  # No weight decay, we use custom L2
        test_fn_grad_clip = optax.clip_by_global_norm(1.0)
        test_fn_zero_nans = optax.zero_nans()
        test_fn_optimizer = optax.chain(test_fn_zero_nans, test_fn_grad_clip, test_fn_optimizer)
        
        # Initialize test function optimizer state
        test_fn_opt_state = test_fn_optimizer.init(eqx.filter(train_state.test_fn, eqx.is_inexact_array))
    else:
        test_fn_optimizer = None
        test_fn_opt_state = None
    
    # Initialize v_theta optimizer state
    v_theta_opt_state = v_theta_optimizer.init(eqx.filter(train_state.v_theta, eqx.is_inexact_array))

    # Training loop
    print("Starting adversarial training...")
    print(f"Epochs: {num_epochs}")
    print(f"V_theta LR: {learning_rate}, Weight decay: {weight_decay}")
    if test_fn_training_enabled:
        print(f"Test_fn LR: {test_fn_learning_rate}, L2 reg: {test_fn_l2_reg}")
    else:
        print("Test function training: DISABLED (not trainable or disabled by user)")
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
        # Split into two phases: test_fn optimization then v_theta optimization
        # ---------------------------------------------
        epoch_losses = []  # store losses to compute an average per epoch
        
        # Phase 1: Train test function for C steps (alternating_frequency) - only if enabled
        test_fn_losses = []
        if test_fn_training_enabled:
            print(f"Epoch {epoch} - Phase 1: Training test function for {alternating_frequency} steps")
            for test_step in range(alternating_frequency):
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

                # Train test function (adversarial: maximize loss + L2 reg)
                train_state, test_fn_opt_state, adversarial_loss, main_loss, l2_penalty = training_step_test_fn(
                    train_state, test_fn_opt_state, test_fn_optimizer, batch_particles, test_fn_l2_reg
                )
                
                test_fn_losses.append(adversarial_loss)
                epoch_losses.append(adversarial_loss)
                
                if test_step % 5 == 0:
                    print(f"  Test_fn Step {test_step} | Adversarial Loss: {adversarial_loss:.6f} | Main Loss: {main_loss:.6f} | L2 Penalty: {l2_penalty:.6f}")
                
                # Log test_fn-specific metrics to wandb
                wandb.log({
                    "test_fn_adversarial_loss": adversarial_loss,
                    "test_fn_main_loss": main_loss,
                    "test_fn_l2_penalty": l2_penalty,
                    "test_fn_step": test_step,
                    "epoch": epoch,
                    "training_phase": "test_fn",
                })
        else:
            print(f"Epoch {epoch} - Phase 1: Skipping test function training (disabled)")

        # Phase 2: Train v_theta for remaining steps (or all steps if test_fn disabled)
        v_theta_steps = steps_per_epoch - (alternating_frequency if test_fn_training_enabled else 0)
        v_theta_losses = []
        print(f"Epoch {epoch} - Phase 2: Training v_theta for {v_theta_steps} steps")
        for v_step in range(v_theta_steps):
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

            # Train velocity field (minimize loss)
            train_state, v_theta_opt_state, v_theta_loss = training_step_v_theta(
                train_state, v_theta_opt_state, v_theta_optimizer, batch_particles
            )
            
            v_theta_losses.append(v_theta_loss)
            epoch_losses.append(v_theta_loss)
            
            if v_step % 10 == 0:
                print(f"  V_theta Step {v_step} | Loss: {v_theta_loss:.6f}")
            
            # Log v_theta-specific metrics to wandb
            wandb.log({
                "v_theta_loss": v_theta_loss,
                "v_theta_step": v_step,
                "epoch": epoch,
                "training_phase": "v_theta",
            })

        # Compute average loss over the epoch for logging/early-stopping etc.
        current_loss = jnp.mean(jnp.stack(epoch_losses))
        avg_test_fn_loss = jnp.mean(jnp.stack(test_fn_losses)) if test_fn_losses else 0.0
        avg_v_theta_loss = jnp.mean(jnp.stack(v_theta_losses)) if v_theta_losses else 0.0

        # Logging
        if epoch % log_interval == 0:
            elapsed_time = time.time() - start_time
            if test_fn_training_enabled:
                print(f"Epoch {epoch:4d} | Avg Loss: {current_loss:.6f} | Test_fn: {avg_test_fn_loss:.6f} | V_theta: {avg_v_theta_loss:.6f} | Time: {elapsed_time:.1f}s")
                print(f"          | Test_fn steps: {len(test_fn_losses)} | V_theta steps: {len(v_theta_losses)}")
            else:
                print(f"Epoch {epoch:4d} | Avg Loss: {current_loss:.6f} | V_theta: {avg_v_theta_loss:.6f} | Time: {elapsed_time:.1f}s")
                print(f"          | V_theta steps: {len(v_theta_losses)} (Test_fn training disabled)")
            
            # Log epoch-level metrics to wandb
            wandb_metrics = {
                "epoch": epoch,
                "avg_loss": current_loss,
                "avg_v_theta_loss": avg_v_theta_loss,
                "v_theta_steps": len(v_theta_losses),
                "elapsed_time": elapsed_time,
                "steps_per_epoch": steps_per_epoch,
                "test_fn_training_enabled": test_fn_training_enabled,
            }
            
            # Only log test function metrics if training is enabled
            if test_fn_training_enabled:
                wandb_metrics.update({
                    "avg_test_fn_loss": avg_test_fn_loss,
                    "test_fn_steps": len(test_fn_losses),
                })
            
            wandb.log(wandb_metrics)
            
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

    final_loss, _, _ = loss_fn(
        v_theta=train_state.v_theta,
        particles=test_particles,
        time_derivative_log_density=train_state.annealed_distribution.unnormalised_time_derivative,
        score_fn=train_state.annealed_distribution.score_fn,
        test_fn=train_state.test_fn,
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



