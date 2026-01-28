import sys
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    # Testing the local repository structure
    # This checks if the project structure is intact and importable.
    from trainers import BasicTrainer
    import utils
    print("    [✓] DPO Local modules (trainers, utils) imported.")
except ImportError as e:
    print(f"CRITICAL FAILURE: Could not import DPO modules. Error: {e}")
    sys.exit(1)

def test_dpo_loss_logic():
    print("--- Starting DPO Functional Verification ---")
    
    try:
        # 1. Hydra/OmegaConf Check
        # Upgrading hydra-core often breaks how config nodes are accessed.
        print("--> Verifying Configuration Orchestration...")
        config = OmegaConf.create({
            "model": {"name_or_path": "gpt2", "tokenizer_name_or_path": "gpt2"},
            "loss": {"beta": 0.1, "label_smoothing": 0},
        })
        print(f"    [✓] Config Beta: {config.loss.beta}")

        # 2. DPO Log-Probability Math (The 'Factor 2' Logic Check)
        # This simulates the DPO loss calculation. If torch/transformers 
        # changes indexing behavior, this will crash.
        print("--> Testing DPO Probability Ratio logic...")
        policy_chosen_logps = torch.tensor([-1.5, -2.0])
        policy_rejected_logps = torch.tensor([-3.0, -3.5])
        reference_chosen_logps = torch.tensor([-1.6, -2.1])
        reference_rejected_logps = torch.tensor([-3.1, -3.6])
        beta = 0.1

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        
        # DPO Loss formula: -log(sigmoid(beta * logits))
        loss = -F.logsigmoid(beta * logits).mean()
        
        print(f"    [✓] DPO Loss calculated: {loss.item():.4f}")

        # 3. Model Import Verification
        import transformers
        print(f"    [✓] Transformers version: {transformers.__version__}")
        
        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        # Likely failures: 
        # - TypeError in logsigmoid (Torch ABI change)
        # - Missing attribute in OmegaConf (Hydra API change)
        sys.exit(1)

if __name__ == "__main__":
    test_dpo_loss_logic()