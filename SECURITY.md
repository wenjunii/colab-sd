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

## Colab GPU runtime: known legacy dependencies

`colab_setup.py` embeds a separate runtime setup in the generated notebook. It still installs Diffusers 0.24.0, Transformers 4.39.3, Hugging Face Hub 0.25.2, and Accelerate 0.30.1 for compatibility with the pinned upstream StreamDiffusion wrapper. The runtime also retains Colab's installed CUDA build of PyTorch, which may be older than the CPU test version. These versions are not security-current. A clean Dependabot view for the test manifest must not be interpreted as a security audit of the Colab runtime.

The bridge uses SD-Turbo and TinyVAE; it does not offer a client control for selecting arbitrary model repositories or enabling remote model code. Continue to use trusted model sources and do not load untrusted Python, pickle/checkpoint files, tokenizers, or model repositories into this runtime. Model-source restrictions reduce exposure but do not patch vulnerable dependencies.

The remaining GPU migration must update the dependency stack together, verify model loading and non-flat image output in both generation modes, check prompt blending and denoise strength, and validate both TensorRT UNets against native outputs. Test-only dependency updates do not establish GPU compatibility. The last GPU validation and its exact software versions are recorded in the README.
