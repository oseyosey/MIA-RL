#!/usr/bin/env python3

# Must set env vars before importing any other modules
import os
import logging

# Now import other modules
import torch
from typing import List
import json
from pathlib import Path
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

class BleurtEvaluator:
    def __init__(self, checkpoint: str = "lucadiliello/BLEURT-20", device: str = None,
                 length_penalty: str = "none", length_threshold: float = 1.5):
        """
        Initialize the BLEURT evaluator with the specified checkpoint.
        Args:
            checkpoint (str): The BLEURT checkpoint to use. Default is lucadiliello/BLEURT-20.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
                                  If None, will use CUDA if available.
            length_penalty (str): Type of length penalty to apply. Options:
                - "none": No length penalty
                - "ratio": Simple length ratio (ref_len/output_len, capped at 1)
                - "sqrt": Square root of ratio (sqrt(ref_len/output_len), capped at 1)
                - "log": Logarithmic penalty (log(1 + ref_len)/log(1 + output_len), capped at 1)
            length_threshold (float): Only apply length penalty when output length exceeds
                                    reference length by this factor. Default is 1.5 (50% longer).
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.length_penalty = length_penalty.lower()
        self.length_threshold = length_threshold
        
        if self.length_penalty not in ["none", "ratio", "sqrt", "log"]:
            raise ValueError(f"Invalid length penalty type: {length_penalty}")
        
        # Load model and tokenizer
        self.config = BleurtConfig.from_pretrained(checkpoint)
        self.model = BleurtForSequenceClassification.from_pretrained(checkpoint).to(self.device)
        self.tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _compute_length_penalty(self, reference: str, candidate: str) -> float:
        """
        Compute length penalty factor based on reference and candidate lengths.
        Returns a value between 0 and 1, where 1 means no penalty.
        """
        ref_len = len(reference.split())
        out_len = len(candidate.split())
        
        # No penalty if output is shorter than threshold * reference length
        if out_len <= self.length_threshold * ref_len:
            return 1.0
            
        ratio = ref_len / out_len
        
        if self.length_penalty == "none":
            return 1.0
        elif self.length_penalty == "ratio":
            return min(1.0, ratio)
        elif self.length_penalty == "sqrt":
            return min(1.0, (ratio) ** 0.5)
        elif self.length_penalty == "log":
            return min(1.0, torch.log(torch.tensor(1 + ref_len)) / torch.log(torch.tensor(1 + out_len)))
        
    def compute_scores(self, reference: str, candidates: List[str]) -> List[float]:
        """
        Compute BLEURT scores for multiple candidates against a single reference.
        
        Args:
            reference (str): The reference text to compare against
            candidates (List[str]): List of candidate texts to evaluate
            
        Returns:
            List[float]: BLEURT scores for each candidate
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        # Create references list of same length as candidates
        references = [reference] * len(candidates)
        
        # Tokenize inputs
        inputs = self.tokenizer(references, candidates, padding='longest', return_tensors='pt')
        
        # Move inputs to correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute base scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            base_scores = outputs.logits.flatten().cpu().tolist()
        
        # Apply length penalty to each score
        scores = [
            base_score * self._compute_length_penalty(reference, candidate)
            for base_score, candidate in zip(base_scores, candidates)
        ]
            
        return scores
    
    def evaluate_with_details(self, reference: str, candidates: List[str]) -> dict:
        """
        Evaluate candidates and return detailed results dictionary.
        
        Args:
            reference (str): The reference text
            candidates (List[str]): List of candidate texts
            
        Returns:
            dict: Dictionary containing reference, candidates and their scores
        """
        scores = self.compute_scores(reference, candidates)
        
        return {
            "reference": reference,
            "evaluations": [
                {"candidate": cand, "score": float(score)}
                for cand, score in zip(candidates, scores)
            ]
        }
    
    def save_results(self, results: dict, output_file: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (dict): Results dictionary from evaluate_with_details
            output_file (str): Path to save results in JSON format
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
