FROM condaforge/miniforge3:latest

# Set working directory
WORKDIR /app

# Install essential build tools and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file first to leverage Docker layer caching
COPY environment.yml .

# Install all conda and pip dependencies into the base environment
RUN conda env update -n base -f environment.yml && conda clean -afy

# Copy the rest of the application source code
COPY . .

# Install the package itself in standard mode without re-installing dependencies
RUN conda run -n base pip install --no-deps .

# Ensure conda bin is in PATH
ENV PATH /opt/conda/bin:$PATH

# Set entrypoint to run under the conda base environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "pyseqrna"]
CMD ["--help"]
