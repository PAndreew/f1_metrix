# Step 1: Base Image
# Use the official Miniconda3 image from Continuum Analytics. It comes with Conda pre-installed.
FROM continuumio/miniconda3

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Create the Conda Environment
# Copy the environment definition file first. This leverages Docker's layer caching.
# The environment will only be rebuilt if environment.yml changes.
COPY environment.yml .

# Create the conda environment from the yml file.
# This is the equivalent of `conda env create -f environment.yml` on your local machine.
RUN conda env create -f environment.yml

# Step 4: Copy Application Code
# Copy the model script and the database into the container's working directory.
COPY model.py .
COPY f1db.db .

# Step 5: Define the Command to Run
# This is the crucial part. We use `conda run` to execute the command
# *inside* the specific environment we just created ('pymc_env').
CMD ["conda", "run", "-n", "pymc_env", "python", "model.py"]