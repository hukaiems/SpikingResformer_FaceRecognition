# 1 Base image: Python 3.10 slim (clean & fast)
FROM python: 3.10

# 2. Set working directory 
WORKDIR /app

# Copy ur project files into the container
COPY . .

# Install dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Run the main script
