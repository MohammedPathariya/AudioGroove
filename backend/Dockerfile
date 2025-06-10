# Use a standard, slim Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install the wget system package
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Create the directories for our model and vocab files
RUN mkdir -p /code/training/checkpoints/lstm_enhanced
RUN mkdir -p /code/data/processed

# Download the large model and vocab files directly into the image
RUN wget -O /code/training/checkpoints/lstm_enhanced/best_epoch_03.pt "https://huggingface.co/pathariyamohammed/audiogroove-lstm/resolve/main/best_epoch_03.pt"
RUN wget -O /code/data/processed/vocab_full_history.jsonl "https://huggingface.co/pathariyamohammed/audiogroove-lstm/resolve/main/vocab_full_history.jsonl"

# Copy the rest of your backend application code
COPY . /code/

# Tell Hugging Face what port the app will run on
EXPOSE 7860

# --- THE ONLY CHANGE IS THIS LAST LINE ---
# Use the "shell form" of CMD and hardcode the exposed port (7860)
CMD gunicorn app:app --bind "0.0.0.0:7860" --timeout 120