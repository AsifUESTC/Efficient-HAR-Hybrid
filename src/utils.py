import torch
import logging
import sys
import gc

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Configure logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
