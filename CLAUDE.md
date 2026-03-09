# CLAUDE.md

## Cluster environment checks

- On every review or work session, first check the current environment and determine whether you are on the cluster head node or a compute node.
- If you are on the school cluster head node, immediately activate the course environment before doing project work:

```bash
source /opt/conda/bin/activate mls
```

- Use the head node for repository management and lightweight setup tasks only, including download, `git pull`, and `git push`.
- Do not run tests on the head node. The head node does not provide the GPU environment required for testing.

## Moving to a compute node

- Use the following command to enter a compute node for GPU work and testing:

```bash
srun -p Teaching --gres=gpu:1 --mem=16G --pty bash
```

- Running `exit` from the compute node returns you to the head node.

## Testing requirements

- All tests must be run on a compute node, not on the head node.
- Before running any test on a compute node, activate the `mls` conda environment first:

```bash
source /opt/conda/bin/activate mls
```
