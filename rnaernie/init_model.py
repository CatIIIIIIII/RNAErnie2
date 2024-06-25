from transformers import AutoConfig, AutoModelForMaskedLM

# Path to the folder containing the pretrained model weights and configuration
folder_path = 'outputs/init'

# Load the configuration
config = AutoConfig.from_pretrained(folder_path)

# Initialize the model with the configuration
# This will create a model with random weights
model = AutoModelForMaskedLM.from_config(config)
model.save_pretrained('outputs/init')
# Print model summary to verify
print(model)

# Now you have a randomly initialized model based on the configuration
