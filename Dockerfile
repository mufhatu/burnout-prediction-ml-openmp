FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project
COPY . .

# Build project
RUN cmake -S . -B build -DBUILD_TESTING=OFF \
    && cmake --build build

WORKDIR /app/build

# Run program
CMD ["./main"]