# LFM2.5-VL-1.6B Test Environment

This folder contains a simple setup to test the `LiquidAI/LFM2.5-VL-1.6B` model.

## Prerequisites

- [Python](https://www.python.org/downloads/) (3.10+ recommended)
- [Git](https://git-scm.com/downloads) (Required for installing the specific transformers version)

## Setup

1.  **Open a terminal in this folder.**
    `c:\Users\Johannes\Documents\KickelToTheCock\world2data\testenvironment\LFM2_5_VL_1_6B`

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **Linux/Mac:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs a specific version of transformers directly from GitHub as required by the model documentation.*

## How to Run

1.  **Configure Input:**
    -   Open `inputs/config.json`.
    -   You can change the `prompt`.
    -   To use an online image, set `image_url`.
    -   To use a local image, set `local_image_path` (e.g., `"inputs/my_image.jpg"`) and `image_url` to `null`.

2.  **Run the script:**
    ```bash
    python run_model.py
    ```

The model will download (the first time), process the image defined in the config, and print the description.

## Running with Video

1.  **Place your video file** (e.g., `my_video.mp4`) in the `inputs/videos` folder.
2.  **Run the video script:**
    ```bash
    python run_video.py
    ```
    The script automatically picks the first video found in that folder, extracts 5 evenly spaced frames, and asks the model to describe the video.
