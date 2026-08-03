# 🎵 AudioGroove: An AI Music Composer

<div align="center">

[![Live App](https://img.shields.io/badge/Live%20Frontend-▲%20Vercel-000000?style=for-the-badge&logo=vercel)](https://audiogroove.vercel.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Live%20Backend-Space-yellow?style=for-the-badge&logo=hugging-face)](https://huggingface.co/spaces/pathariyamohammed/audiogroove-hf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

**AudioGroove is an AI-powered music generation system that learns from MIDI files to compose new, original musical sequences. It leverages a deep LSTM model enhanced with a self-attention mechanism to understand and replicate complex musical patterns.**

---

## 🚀 Experience the Live Demo

Generate your own unique compositions with a single click. See what the AI creates!

**[https://audiogroove.vercel.app/](https://audiogroove.vercel.app/)**

---

## 💡 My Motivation

As someone who's always been fascinated by both the structured logic of code and the soulful expression of music, I kept asking myself a question: could a machine do more than just rearrange notes? Could it actually learn the *feeling*, the structure, and the creative spark that makes a piece of music memorable?

AudioGroove is my answer to that question. This project started as a personal challenge—not just to build another sequence generator, but to see if I could complete the entire journey from raw data to a live, interactive web application. It was about diving deep into the MLOps lifecycle, wrestling with real-world deployment problems, and ultimately creating something that anyone, anywhere, could use to experience a touch of AI-driven creativity.

## ✨ Key Features

- **AI-Powered Composition:** Provides the application path for MIDI composition while reproducible pilot training and benchmarking are completed.
- **Creative Seeding:** You can upload your own `.mid` file to give the AI a starting point, influencing the melody and style of the output.
- **Freestyle Generation:** If you don't provide a seed, the backend will pick one at random, leading to surprising and unique compositions.
- **Interactive & Modern UI:** A clean and responsive web interface built with vanilla HTML, CSS, and JavaScript, ensuring a fast and lightweight user experience.
- **Robust Decoupled Architecture:** A production-ready system with separate frontend and backend deployments for better scalability and maintainability.

---

## 🛠️ Tech Stack & Architecture

AudioGroove is built with a modern, decoupled architecture, with each component chosen for its specific strengths in a production environment.

**Frontend:**
- **Technology:** Vanilla HTML, CSS, JavaScript (no frameworks for a lean, fast-loading experience).
- **Deployment:** [**Vercel**](https://vercel.com/) for high-performance static site hosting and seamless continuous deployment from Git.

**Backend:**
- **Framework:** [**Flask**](https://flask.palletsprojects.com/) served by [**Gunicorn**](https://gunicorn.org/), providing a lightweight yet powerful Python API.
- **Deployment:** [**Hugging Face Spaces**](https://huggingface.co/spaces) which offers the necessary free CPU/RAM resources to run the ML model effectively.
- **Containerization:** [**Docker**](https://www.docker.com/) to create a consistent and reproducible runtime environment for the server.

**Machine Learning:**
- **Core Model:** A PyTorch-based LSTM with a Multi-Head Self-Attention layer.
- **Data Processing:** Dask for bounded parallel MIDI ETL and `music21` or `mido` for MIDI parsing and feature extraction.
- **Experiment Tracking:** MLflow for reproducible local training runs, metrics, checkpoints, and benchmark artifacts.
- **Artifact Hosting:** [**Hugging Face Hub**](https://huggingface.co/docs/hub/index) to store the large model checkpoint (`.pt`) and vocabulary file (`.jsonl`), keeping the source code repository lightweight.

### System Architecture Diagram

[ User on Vercel Frontend ]
|
| (HTTPS API Request)
V
[ Hugging Face Space (Docker Container) ]
|
|---[ Gunicorn Server ]
|      |
|      +---[ Flask App (app.py) ]
|             |
|             +---[ PyTorch Model ] --> Generates Music
|
| (Returns generated .mid file)
V
[ User Downloads Composition ]


---

## ⚙️ My Process: From Data to Deployment

My journey with this project followed a complete machine learning lifecycle:

1.  **Data Collection & Preparation:** The current verified audit uses 10 local seed MIDI files. The next controlled stage is an approximately 250-song LMDClean pilot with deterministic source selection, source-level splits, non-destructive quarantine, bounded chunks, and recorded parser failures. The larger corpus remains a later scale-up phase.

2.  **Vocabulary Building:** The vocabulary and representation are being rebuilt as part of the bounded pilot so every model comparison uses the same versioned preprocessing artifacts.

3.  **Model & Training:** The pilot will compare a compact LSTM, GRU, and compact Transformer on the same frozen source split and training budget. Dask will provide bounded preprocessing and MLflow will track the runs. The existing attention model is a prototype, not yet a verified winner.

4.  **Generation Logic:** The generation path supports autoregressive sampling with temperature and top-k controls. Model-quality and generation-validity claims remain pending the controlled pilot benchmark.

---

## 🧗 Challenges & Deployment Battles

Deploying a machine learning app on a free budget is a true test of problem-solving. Here are the battles I fought and won:

- **Challenge:** **The GitHub 100 MB Limit.** Both my model checkpoint and final vocabulary file were massive, far exceeding GitHub's file size limit.
- **Solution:** **Decoupling Large Artifacts.** I adopted a standard MLOps practice by hosting all large files on the **Hugging Face Hub**. My deployment server was then configured to download these artifacts during its build step using `wget`, keeping my Git repository lean and focused on code.

- **Challenge:** **The Render Free Tier Timeout.** My first deployment attempt on Render kept failing. The logs showed a `WORKER TIMEOUT` because the music generation was too resource-intensive for the free plan's CPU and 30-second time limit.
- **Solution:** **Migrating to the Right Tool for the Job.** I pivoted and migrated the backend from Render to **Hugging Face Spaces**. Spaces are specifically designed for hosting ML apps and provide a much more generous free tier of CPU/RAM, which completely solved the timeout issues.

- **Challenge:** **The Docker `ModuleNotFoundError`.** After containerizing the app, it failed to boot, complaining that it couldn't find my custom Python modules (like `models` or `utils`).
- **Solution:** **Creating a Self-Contained Deployment Package.** I refactored the project to create a clean, self-contained deployment folder. This folder included the Flask app, the Dockerfile, and the entire `src` directory, ensuring that the Docker container had everything it needed to run, finally resolving the import errors.

---

## 🚧 Limitations & The Road Ahead

- **Limitation:** Performance on the free-tier hardware means that generating very long or complex pieces can still be slow.
- **Limitation:** While the model captures patterns well, it doesn't have a formal understanding of music theory. This can sometimes result in compositions that are musically interesting but lack traditional long-form structure.

- **Future Work:**
  - **Smarter Models:** I'm excited to experiment with more advanced architectures like Transformers, which could capture longer-range dependencies in the music.
  - **Going GPU:** Deploying the model on a GPU-enabled service would cut generation time from minutes to seconds.
  - **User-Driven Creativity:** I plan to add frontend controls that allow users to directly influence the generation by tweaking parameters like `temperature` and `top-k` sampling.

---

## License

This project is licensed under the **MIT License**. Feel free to explore, fork, and build upon it!
