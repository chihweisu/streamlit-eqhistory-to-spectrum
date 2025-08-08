# Streamlit Response Spectrum Generator (streamlit-eqhistory-to-spectrum)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-eqhistory-to-spectrum-upgfbumd4xcjrrtbc9jeja.streamlit.app/)
[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**🚀 [Try the Live Demo](https://app-eqhistory-to-spectrum-upgfbumd4xcjrrtbc9jeja.streamlit.app/)**

This is an interactive web app built with Streamlit, designed for structural engineers and researchers to quickly convert earthquake time-history data into structural response spectra, with advanced scaling and comparison features.

## 📚 Learn More

For a detailed walkthrough, technical insights, and the story behind this tool, check out my comprehensive article:

**📝 ["From Raw Seismic Data to Insightful Spectra in Seconds"](https://medium.com/@jj19960130/from-raw-seismic-data-to-insightful-spectra-in-seconds-54a44f57f762)**

---

## Key Features

* **Versatile Spectrum Analysis**: Simultaneously calculates and plots three types of response spectra: **Absolute Acceleration (Sa)**, **Pseudo-Spectral Velocity (Sv)**, and **Spectral Displacement (Sd)**.
* **Parametric Configuration**: Users can freely adjust core analysis parameters such as damping ratio (ξ), period range, and the number of period points.
* **High-Precision Calculation**:
    * Built-in **time-history interpolation** allows for a custom analysis time step (dt) to ensure the accuracy of short-period responses.
    * Supports **logarithmic period sampling** to more effectively capture response details in the short-period range.
* **Design Code Spectrum Overlay**: Allows input of `SDS` and `SD1` parameters to directly overlay the design response spectrum from current seismic codes on the Sa plot for easy comparison.
* **Intelligent Scaling Factor Calculation**:
    * Users can define a target period `T`.
    * The app automatically calculates a scaling factor to ensure the response spectrum meets code requirements within the `0.2T` to `1.5T` range (point-to-point ≥ 90% of the design spectrum & the average value of the spectrum ≥ 100% of the average value of the design spectrum).
    * Visually presents the scaled response spectrum and the highlighted scaling range.
* **Data Export**:
    * One-click download of the **scaled time-history** as a text file.
    * All response spectrum plots can be downloaded as high-quality `.png` images.
* **Secure Login Mechanism**: Integrates `st.secrets` to provide a simple password-protected login page, securing access to the application.
* **User-Friendly Interface**:
    * Features a real-time update mechanism; adjusting display options (like axis scale or design spectrum) does not require re-running the analysis, ensuring a smooth interactive experience.
    * Intelligently suggests an appropriate analysis `dt` and syncs the value across the UI.

---

## Getting Started

### Prerequisites

* Python 3.8+

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Pulla87/streamlit-eqhistory-to-spectrum.git
    cd streamlit-eqhistory-to-spectrum
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

1.  **Set Up Your Password**:
    * In the project's root directory, create a new folder named `.streamlit`.
    * Inside the `.streamlit` folder, create a new file named `secrets.toml`.
    * Add the following content to `secrets.toml` and set your password:
        ```toml
        APP_PASSWORD = "your_secret_password"
        ```

2.  **Launch the App**:
    ```bash
    streamlit run your_app_file_name.py
    ```
    > Replace `your_app_file_name.py` with the name of your main Python script.

---

## How to Use

1.  After launching the app, enter the password you configured in `secrets.toml` on the login page.
2.  On the main screen, click "Browse files" to upload your earthquake time-history file.
    * **File Format**: A plain text file (`.txt`, `.csv`) with two columns of numerical data and no header. The first column is time (sec), and the second is acceleration (gal).
3.  In the left sidebar, configure the **core parameters** (e.g., damping ratio, period range).
4.  Click the **"Calculate and Plot Response Spectra"** button on the main page and wait for the analysis to complete.
5.  Once finished, the three response spectrum plots will be displayed below.
6.  You can adjust the **display settings** (e.g., axis scale) or the target period for **time-history scaling** in the sidebar at any time, and the plots will update instantly.

---

## Deployment

This project is configured to use `st.secrets` and can be deployed directly to [Streamlit Community Cloud](https://streamlit.io/cloud).

1.  Push your project to a public GitHub repository.
2.  When deploying, go to "Advanced settings..." and paste your password in TOML format into the "Secrets" text box:
    ```toml
    APP_PASSWORD = "your_secret_password"
    ```
3.  Click "Deploy!".

---

## License

This project is licensed under the [MIT License](LICENSE).
