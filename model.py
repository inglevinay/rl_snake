import torch
import torch.nn as nn
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'): # Let's default to the new filename
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        
        if os.path.exists(file_path):
            # --- THIS IS THE FIX ---
            # 1. Load the entire checkpoint dictionary
            checkpoint = torch.load(file_path)
            
            # 2. Extract ONLY the model's state_dict from the checkpoint
            model_state = checkpoint['model_state_dict']

            # 3. Load that state_dict into the model
            self.load_state_dict(model_state)
            
            self.eval() # Set the model to evaluation mode for playing
            print(f"Loaded model state from {file_path}")
        else:
            print(f"Could not find model file at {file_path}")