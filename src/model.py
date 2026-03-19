"""
model.py -- BERT Classifier for Resume Role Prediction
========================================================

This module defines the neural network architecture:
    BERT (pretrained) + Classification Head (custom)

Deep Learning Concepts Demonstrated:
1. Transfer Learning: Using pretrained BERT weights as a starting point
2. Classification Head: Linear layer mapping embeddings to class logits
3. Dropout: Regularization to prevent overfitting
4. Activation Functions: ReLU (implicit in BERT), Softmax (in loss)
5. Forward Pass: The computation graph from input to output

Architecture:
    Input (input_ids, attention_mask)
        |
        v
    BERT Encoder (12 transformer layers, 768-dim hidden)
        |
        v
    [CLS] token embedding (768-dim vector)
        |
        v
    Dropout (p=0.3) -- Regularization
        |
        v
    Linear (768 -> num_classes) -- Classification head
        |
        v
    Logits (raw scores for each class)

Why this architecture works:
- BERT has already learned deep language understanding from pretraining
- The [CLS] token acts as a "summary" of the entire input
- We only need a thin classification head on top
- Fine-tuning adjusts BERT's weights slightly for our specific task

Author: AI Resume Analyzer Project
"""

import torch
import torch.nn as nn
from transformers import BertModel


class ResumeClassifier(nn.Module):
    """
    BERT-based classifier for predicting job roles from resumes.
    
    This is a classic Transfer Learning setup:
    1. Start with pretrained BERT (learned from Wikipedia + BookCorpus)
    2. Add a task-specific classification head
    3. Fine-tune both BERT and the head on our resume data
    
    nn.Module is the base class for ALL neural networks in PyTorch.
    Every custom model must:
    - Inherit from nn.Module
    - Define layers in __init__()
    - Implement forward() for the computation
    """
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.3,
                 model_name: str = "bert-base-uncased"):
        """
        Initialize the model architecture.
        
        Args:
            num_classes: Number of job role categories (5 in our case)
            dropout_rate: Probability of zeroing neurons during training
                         0.3 = 30% of neurons are randomly disabled per forward pass
                         Why: Forces the network to not rely on any single neuron,
                         making it more robust (regularization technique)
            model_name: Pretrained BERT variant to use
        """
        # super().__init__() is REQUIRED for nn.Module
        # It registers this class with PyTorch's parameter tracking system
        # Without it, model.parameters() won't find any weights to optimize
        super(ResumeClassifier, self).__init__()
        
        # ================================================================
        # LAYER 1: Pretrained BERT Encoder
        # ================================================================
        # BertModel loads the pretrained weights (~440MB download on first run)
        # 
        # BERT architecture inside:
        #   - Embedding layer: token IDs -> 768-dim vectors
        #   - 12 Transformer encoder layers, each with:
        #     - Multi-Head Self-Attention (12 heads)
        #     - Feed-Forward Network (768 -> 3072 -> 768)
        #     - Layer Normalization
        #     - Residual Connections
        #   - Total: ~110 million parameters
        #
        # The output is a 768-dim vector for each input token
        # We only use the [CLS] token's vector for classification
        self.bert = BertModel.from_pretrained(model_name)
        
        # ================================================================
        # LAYER 2: Dropout (Regularization)
        # ================================================================
        # During training: Randomly zeroes 30% of the 768-dim [CLS] vector
        # During inference: No dropout (all neurons active, outputs scaled)
        #
        # Why Dropout works:
        # - Prevents "co-adaptation" where some neurons become too dependent
        #   on specific other neurons
        # - Acts like training many smaller networks and averaging them
        # - Especially important here because:
        #   a) BERT is very large (110M params)
        #   b) Our dataset is small (10 resumes)
        #   c) Without regularization, the model would memorize training data
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # ================================================================
        # LAYER 3: Classification Head (Linear Layer)
        # ================================================================
        # Linear(in_features, out_features) performs: y = Wx + b
        #   - W: Weight matrix of shape [num_classes, 768]
        #   - b: Bias vector of shape [num_classes]
        #   - x: [CLS] embedding of shape [768]
        #   - y: Output logits of shape [num_classes]
        #
        # Why 768 -> num_classes?
        #   - 768 is BERT's hidden size (each token is a 768-dim vector)
        #   - num_classes is our output size (5 job roles)
        #   - This single linear layer maps BERT's representation space
        #     directly to our classification space
        #
        # Why not add more layers?
        #   - BERT already provides rich features
        #   - More layers = more parameters = more overfitting risk
        #   - Simple heads work best with pretrained transformers
        #   - Research shows diminishing returns from deeper heads
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Store config for reference
        self.num_classes = num_classes
        
        print(f"[Model] ResumeClassifier initialized:")
        print(f"   BERT: {model_name} ({sum(p.numel() for p in self.bert.parameters()):,} params)")
        print(f"   Classifier: {self.bert.config.hidden_size} -> {num_classes}")
        print(f"   Dropout: {dropout_rate}")
        print(f"   Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Transform input tokens into class predictions.
        
        This defines the COMPUTATION GRAPH that PyTorch uses for:
        1. Forward pass: input -> output (making predictions)
        2. Backward pass: output -> gradients (computing how to update weights)
        
        PyTorch's autograd automatically builds the backward graph
        from this forward definition. This is called
        "automatic differentiation" — the key to backpropagation.
        
        Args:
            input_ids: Token IDs from BERT tokenizer
                      Shape: [batch_size, seq_length]
                      Example: [[101, 3698, 4083, 102, 0, ...], ...]
                      
            attention_mask: Binary mask indicating real vs. padded tokens
                          Shape: [batch_size, seq_length]
                          Example: [[1, 1, 1, 1, 0, ...], ...]
        
        Returns:
            logits: Raw prediction scores for each class
                   Shape: [batch_size, num_classes]
                   Example: [[-0.5, 1.2, 0.3, -0.8, 0.1], ...]
                   
                   NOTE: These are LOGITS (raw scores), NOT probabilities.
                   CrossEntropyLoss applies LogSoftmax internally.
                   For probabilities during inference, use:
                   probs = torch.softmax(logits, dim=1)
        """
        # ============================================================
        # Step 1: Pass through BERT encoder
        # ============================================================
        # BERT returns a complex output object with multiple attributes:
        #   - last_hidden_state: All token embeddings [batch, seq_len, 768]
        #   - pooler_output: [CLS] token embedding [batch, 768]
        #                    (passed through a linear layer + tanh)
        #
        # We use pooler_output because:
        #   - It's specifically designed for classification tasks
        #   - BERT was pretrained with a "Next Sentence Prediction" task
        #     that used this same pooled output
        #   - It captures the "meaning" of the entire input sequence
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract the [CLS] token's pooled representation
        # Shape: [batch_size, 768]
        pooled_output = bert_output.pooler_output
        
        # ============================================================
        # Step 2: Apply dropout (only active during training)
        # ============================================================
        # model.train()  -> dropout is ON  (randomly zeros 30% of values)
        # model.eval()   -> dropout is OFF (all values pass through)
        #
        # This is why we always call model.train() before training
        # and model.eval() before inference
        pooled_output = self.dropout(pooled_output)
        
        # ============================================================
        # Step 3: Classification head
        # ============================================================
        # Linear transformation: [batch, 768] -> [batch, num_classes]
        # This maps the rich 768-dim BERT representation to our
        # 5-dimensional class space
        logits = self.classifier(pooled_output)
        
        return logits
    
    def freeze_bert(self):
        """
        Freeze BERT's parameters so only the classification head trains.
        
        Use case: When you have very little data and want to prevent
        BERT from "forgetting" its pretrained knowledge.
        
        How freezing works:
        - param.requires_grad = False tells PyTorch:
          "Don't compute gradients for this parameter"
        - Since gradients drive weight updates (backpropagation),
          frozen parameters stay at their pretrained values
        - Only the unfrozen classifier head gets updated
        
        Tradeoff:
        - Frozen: Safer, less overfitting, but less task adaptation
        - Unfrozen: Better task performance, but needs more data
        """
        for param in self.bert.parameters():
            param.requires_grad = False
        print("[Model] BERT layers frozen. Only classifier head will train.")
    
    def unfreeze_bert(self, num_layers: int = None):
        """
        Unfreeze BERT layers for fine-tuning.
        
        Strategy: Gradually unfreeze from top layers down.
        Top layers capture task-specific features (should be fine-tuned).
        Bottom layers capture general language features (can stay frozen).
        
        Args:
            num_layers: Number of top encoder layers to unfreeze.
                       None = unfreeze all layers.
                       
        Example:
            model.unfreeze_bert(num_layers=4)  
            # Unfreezes layers 8-11 (top 4 of 12)
            # Keeps layers 0-7 frozen (general language understanding)
        """
        if num_layers is None:
            # Unfreeze everything
            for param in self.bert.parameters():
                param.requires_grad = True
            print("[Model] All BERT layers unfrozen for fine-tuning.")
        else:
            # Unfreeze only the top N encoder layers
            total_layers = len(self.bert.encoder.layer)
            for i, layer in enumerate(self.bert.encoder.layer):
                if i >= total_layers - num_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            # Always unfreeze the pooler
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            print(f"[Model] Top {num_layers} BERT layers unfrozen.")


def get_model(num_classes: int = 5, dropout_rate: float = 0.3) -> ResumeClassifier:
    """
    Factory function to create and configure the model.
    
    This is a convenience function that:
    1. Creates the model
    2. Moves it to GPU if available
    3. Returns the model and device info
    
    Args:
        num_classes: Number of classification categories
        dropout_rate: Dropout probability
        
    Returns:
        Configured ResumeClassifier model
    """
    # Check for GPU availability
    # CUDA = NVIDIA GPU, MPS = Apple Silicon GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (training will be slower)")
    
    # Create model and move to device
    model = ResumeClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    return model, device


# ============================================================================
# Quick test -- run this file directly to verify model architecture
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 60)
    
    # Create model
    model, device = get_model(num_classes=5)
    
    # Print full architecture
    print(f"\n{'=' * 60}")
    print("ARCHITECTURE SUMMARY")
    print(f"{'=' * 60}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy data
    print(f"\n{'=' * 60}")
    print("FORWARD PASS TEST")
    print(f"{'=' * 60}")
    
    # Create fake batch: 2 samples, 128 tokens each
    batch_size = 2
    seq_length = 128
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long).to(device)
    
    # Forward pass
    model.eval()  # Set to evaluation mode (disables dropout)
    with torch.no_grad():  # Disable gradient computation (saves memory)
        logits = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"\n   Input shapes:")
    print(f"      input_ids:      {dummy_input_ids.shape}")
    print(f"      attention_mask:  {dummy_attention_mask.shape}")
    print(f"   Output shape:       {logits.shape}")
    print(f"   Output (logits):    {logits}")
    
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"   Probabilities:      {probs}")
    print(f"   Predicted classes:  {torch.argmax(probs, dim=1).tolist()}")
    
    # Test freezing
    print(f"\n{'=' * 60}")
    print("FREEZE/UNFREEZE TEST")
    print(f"{'=' * 60}")
    
    model.freeze_bert()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable after freeze: {frozen_trainable:,}")
    
    model.unfreeze_bert(num_layers=4)
    partial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable after unfreeze top-4: {partial_trainable:,}")
    
    model.unfreeze_bert()
    full_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable after full unfreeze: {full_trainable:,}")
    
    print(f"\n[OK] All model tests passed!")
