import subprocess


def run(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=False)


def main():
    # Install CUDA torch wheels and project deps
    run("pip install -U pip wheel")
    run(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    # Include grounding extra to install spaCy on Colab
    run("pip install -e .[dev,grounding]")
    run("python -m spacy download en_core_web_sm")
    # Smoke tests (non-fatal)
    run("pytest -q || true")


if __name__ == "__main__":
    main()
