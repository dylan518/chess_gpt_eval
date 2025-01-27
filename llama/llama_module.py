###LLAMA Module for gpu:
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from typing import Optional

class BaseLlamaPlayer:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, model_name: str):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name

    def get_llama_response(self, game_state: str, temperature: float) -> Optional[str]:
        prompt = game_state
        # Move input_ids to the same device as the model
        tokenized_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            result = self.model.generate(
                **tokenized_input, max_new_tokens=10, temperature=temperature
            )
        input_ids_tensor = tokenized_input["input_ids"]
        res_sliced = result[:, input_ids_tensor.shape[1]:]
        return self.tokenizer.batch_decode(res_sliced, skip_special_tokens=True)[0]

    def get_move_from_response(self, response: Optional[str]) -> Optional[str]:
        if response is None:
            return None
        # Parse the response to get only the first move
        moves = response.split()
        first_move = moves[0] if moves else None
        return first_move

    def get_move(self, board: str, game_state: str, temperature: float) -> Optional[str]:
        completion = self.get_llama_response(game_state, temperature)
        return self.get_move_from_response(completion)

    def get_config(self) -> dict:
        return {"model": self.model_name}

class LocalLlamaPlayer(BaseLlamaPlayer):
    def __init__(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Changed dtype
            device_map="auto"           # Changed device_map
        )
        super().__init__(tokenizer, model, model_name)

class LocalLoraLlamaPlayer(BaseLlamaPlayer):
    def __init__(self, base_model_id: str, adapter_model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
        model = (
            PeftModel.from_pretrained(base_model, adapter_model_path)
            .merge_and_unload()
            .to(base_model.device)
        )
        super().__init__(tokenizer, model, adapter_model_path)
