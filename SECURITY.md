# Security and dependency scope

## CPU checks

`requirements-test.txt` is used by GitHub Actions and the offline model stand-ins. It uses Torch 2.13.0 and Transformers 5.10.1 to address the 27 Dependabot alerts reported against the previous Torch 2.5.1 and Transformers 4.39.3 pins. Transformers 5.10.1 requires Hugging Face Hub >=1.5,<2; keeping the old Hub 0.25.2 pin prevents installation. FastAPI is upgraded with its compatible Starlette dependency as well. The wider installed-package audit also requires Pillow >=13, filelock >=4.2.1, and setuptools >=84.1.0. Patched Pillow and filelock require Python 3.11, which is now the development and CI baseline.

The workflow installs these requirements with the official PyTorch CPU wheel index in one resolver transaction, runs `pip check` before the tests, and audits the installed environment with `audit_dependencies.py` and `pip-audit`. Future Torch updates therefore also update the wheel selected by CI, without a second version pin in the workflow. The audit reports newly published vulnerabilities as CI failures instead of suppressing alerts.

`pip-audit` cannot look up Torch's local `+cpu` version on PyPI. The audit helper therefore creates an exact inventory of the installed packages, maps only the official Torch `+cpu` build to its base release, and audits that inventory without resolving or installing packages again. Other local builds are rejected. Skipped or missing audit results fail the check so an unaudited package cannot silently pass.

To audit the installed development environment:

```powershell
.\.venv\Scripts\python.exe -m pip install pip-audit==2.10.1
.\.venv\Scripts\python.exe audit_dependencies.py
```

## GPU runtimes: known legacy dependencies

`colab_setup.py` embeds a separate runtime setup in the generated notebook. It still installs Diffusers 0.24.0, Transformers 4.39.3, Hugging Face Hub 0.25.2, and Accelerate 0.30.1 for compatibility with the pinned upstream StreamDiffusion wrapper. The runtime also retains Colab's installed CUDA build of PyTorch, which may be older than the CPU test version. These versions are not security-current. A clean Dependabot view for the test manifest must not be interpreted as a security audit of the Colab runtime.

The bridge uses SD-Turbo and TinyVAE; it does not offer a client control for selecting arbitrary model repositories or enabling remote model code. Continue to use trusted model sources and do not load untrusted Python, pickle/checkpoint files, tokenizers, or model repositories into this runtime. Model-source restrictions reduce exposure but do not patch vulnerable dependencies.

The remaining GPU migration must update the dependency stack together, verify model loading and non-flat image output in both generation modes, check prompt blending and denoise strength, and validate both TensorRT UNets against native outputs. Test-only dependency updates do not establish GPU compatibility. The last GPU validation and its exact software versions are recorded in the README.

TensorRT deployment archives contain executable model graphs and must come from your own trusted runtime. Keep Drive/cache folders private; checksums detect accidental corruption, not a malicious replacement of both archive and manifest. The cache validates hardware/software/model metadata and numerical output before use. Stripped/refit engine caching remains disabled.

## Optional Waypoint runtime

`Waypoint_Colab_Server.ipynb` installs `waypoint_setup.py` dependencies in a separate environment, rather than replacing SD-Turbo's compatibility stack or the notebook kernel. The runtime source is pinned to a Git commit. `waypoint_backend.py` downloads only the pinned official model's configuration/safetensors and the pinned official TAEHV checkpoint; no client-selected model code is loaded. The TAEHV runtime reads a trusted PyTorch checkpoint. Keep model cache folders private and do not substitute untrusted checkpoints.

Waypoint currently requires Torch 2.11.0 upstream, plus its quantization/compiler dependencies. The CPU suite uses a different, newer Torch version. Its dependency audit does not cover this optional GPU environment or establish A100 compatibility. The installer runs `pip check` and a CUDA/import preflight; those are dependency/availability checks, not vulnerability or image-quality checks. Some transitive dependencies remain resolved by upstream constraints. Cloudflared uses the official release download, matching the existing launcher setup.

The new world-mode controls, seed uploads, delivery credit handling and reconnect lifecycle have CPU contract tests. A100 40 GB model loading, compilation and generated video still require a live check. The model is image-conditioned; it does not receive the SD-Turbo prompt controls. The existing single-controller and optional token authentication gates also apply to this backend. The Colab notebook defaults to a blank token, as does the SD-Turbo notebook; anyone with its tunnel URL can attempt to connect.

## Workstation access

`workstation.py setup` creates its own environment rather than modifying system Python. It reuses the same legacy model dependency pins listed above, with a selected CUDA Torch wheel and patched Pillow/filelock/setuptools. The CPU dependency audit does not establish security of this GPU environment.

The workstation launcher requires a persistent random token, stored in `.workstation/access-token` (owner-only permissions on Linux; inherited user-folder permissions on Windows). The client must authenticate before receiving a welcome message, submitting controls, receiving video, or taking the single controller slot. The token is not placed in URLs, server arguments, health responses, snapshots, or server/tunnel logs. It is displayed in the launching terminal for pairing.

Use direct `ws://` only over trusted LAN/VPN connections. For internet use, choose the TLS Cloudflare transport or a VPN; the token does not encrypt plain WebSocket traffic. Cloudflare mode binds to loopback unless explicitly overridden. No firewall or router settings are modified automatically. The existing Colab setup remains compatible with a blank token; it does not gain workstation authentication automatically.

TouchDesigner saves the entered access token in the project/component parameters. Clear it before sharing a paired `.toe` or `.tox` file. Distributed project builds contain an empty token. Only connect to server URLs you trust.
