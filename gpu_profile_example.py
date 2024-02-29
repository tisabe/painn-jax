import jax
import jax.numpy as jnp

@jax.jit
def multiply(a, b):
    print("Test print that should only be printed once")
    return jnp.matmul(a, b)

jax.profiler.start_trace("./slurm/tensorboard", create_perfetto_trace=True)

key = jax.random.PRNGKey(42)

# multiply random matrices 100 times
for _ in range(100):
    key, subkey = jax.random.split(key)
    a = jax.random.normal(subkey, shape=(1000, 1000))
    key, subkey = jax.random.split(key)
    b = jax.random.normal(subkey, shape=(1000, 1000))

    res = multiply(a, b)
    # the following changes the asynchronous dispatch behavior:
    #res.block_until_ready()

print("Done")

jax.profiler.stop_trace()