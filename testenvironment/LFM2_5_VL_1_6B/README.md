# LFM2.5-VL-1.6B Test Environment

This folder contains a simple setup to test the `LiquidAI/LFM2.5-VL-1.6B` model with full GPU support.

## Prerequisites

- [Python](https://www.python.org/downloads/) (3.10+ recommended)
- [Git](https://git-scm.com/downloads)
- **NVIDIA GPU**: Required for reasonable performance.

## Setup

1.  **Open a terminal in this folder.**
    `c:\Users\Johannes\Documents\KickelToTheCock\world2data\testenvironment\LFM2_5_VL_1_6B`

2.  **Create a virtual environment (if not already done):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows (PowerShell):**
        ```powershell
        .\.venv\Scripts\activate
        ```
    -   **Git Bash / Mac / Linux:**
        ```bash
        source .venv/Scripts/activate
        ```

4.  **Install dependencies:**
    **CRITICAL:** We must install PyTorch with CUDA 12.1 (or compatible) support explicitly first, then the rest.
    
    ```bash
    # 1. Install PyTorch with CUDA support (Use cu124 for Python 3.13+)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # 2. Install the rest of the requirements
    pip install -r requirements.txt
    ```

## How to Run

1.  **Video Analysis (GPU Optimized):**
    -   Place your `.mp4` video in `inputs/videos/`.
    -   Run the script:
        ```bash
        python run_video.py
        ```
    -   The script will verify your GPU is active and process the video in chunks.
    -   Output is saved to `outputs/your_video_name.txt`.

2.  **Single Image Test:**
    -   Edit `inputs/config.json`.
    -   Run:
        ```bash
        python run_model.py
        ```
