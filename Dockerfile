FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# First run the training script
RUN python src/finetuning/train.py

# Set default command to run Gradio app
CMD ["python", "src/gradio_chat/main.py"]