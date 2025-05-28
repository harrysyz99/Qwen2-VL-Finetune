import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast # or Qwen2_5_VLCausalLMOutputWithPast
from typing import Optional, List, Union, Tuple, Dict

from .constants import TASK_CONFIG, TASK_TYPES, TASK_TYPE_TO_ID, ID_TO_TASK_TYPE, IGNORE_INDEX

# Determine which base model and output class to use (Qwen2 or Qwen2.5)
# This could be made more dynamic based on the config of the loaded model
# For now, let's default to Qwen2.5-VL as it's newer. Users might use either.
# We will try to make this selection dynamic in the __init__.

class MultiHeadQwenVLOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    def __init__(
        self,
        loss: Optional[torch.FloatTensor] = None,
        logits: Optional[torch.FloatTensor] = None, # This will be the type_logits
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
        rope_deltas: Optional[Tuple[torch.Tensor]] = None,
        # Custom outputs
        type_logits: Optional[torch.FloatTensor] = None,
        type_probs: Optional[torch.FloatTensor] = None,
        classification_outputs: Optional[Dict[str, torch.FloatTensor]] = None,
        regression_output: Optional[torch.FloatTensor] = None,
    ):
        super().__init__(
            loss=loss,
            logits=logits, # Pass type_logits to parent's logits for compatibility if needed
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            rope_deltas=rope_deltas,
        )
        self.type_logits = type_logits
        self.type_probs = type_probs
        self.classification_outputs = classification_outputs
        self.regression_output = regression_output

class MultiHeadQwenVLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration): # Defaulting to Qwen2.5, can be Qwen2VLForConditionalGeneration
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.task_config = TASK_CONFIG
        self.task_types = TASK_TYPES
        self.task_type_to_id = TASK_TYPE_TO_ID
        self.id_to_task_type = ID_TO_TASK_TYPE

        # Type head: predicts the probability for each task in TASK_TYPES
        self.type_head = nn.Linear(self.hidden_size, len(self.task_types))

        # Classification heads: one for each classification task
        self.classification_heads = nn.ModuleDict()
        for task_name, details in self.task_config.items():
            if details["type"] == "classification":
                self.classification_heads[task_name] = nn.Linear(self.hidden_size, details["num_categories"])

        # Regression head: one for regression task (if defined)
        if "regression" in self.task_config:
            self.regression_head = nn.Linear(self.hidden_size, self.task_config["regression"]["output_dim"])
        else:
            self.regression_head = None
        
        # The original lm_head is part of Qwen2_5_VLForConditionalGeneration.
        # We are not using it for the standard LM task in this multi-head setup.
        # It will still exist but its output won't be used directly in our loss.
        # Or, we could set self.lm_head = nn.Identity() if we want to be absolutely sure,
        # but that might affect weight loading if not handled carefully.

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # Custom labels for multi-head training
        task_id_labels: Optional[torch.LongTensor] = None,
        classification_labels: Optional[torch.LongTensor] = None, # Expected shape: [batch_size, num_classification_tasks]
        regression_labels: Optional[torch.FloatTensor] = None, # Expected shape: [batch_size]
    ) -> Union[Tuple, MultiHeadQwenVLOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard Qwen-VL forward pass to get hidden_states
        # This replicates the logic from the original forward method (including monkey-patched parts if this class is used with them)
        # up to the point of getting transformer outputs.
        
        # We need to call the underlying transformer model: self.model
        # The input embedding layer and visual processing is part of the Qwen2_5_VLForConditionalGeneration class, not self.model directly.

        # Step 1: Get inputs_embeds (handles text, image, video modalities)
        # This logic is adapted from Qwen2_5_VLForConditionalGeneration.forward
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
    
        if pixel_values is None and pixel_values_videos is None and inputs_embeds.shape[0] > 0: # check for empty batch
            # This dummy pass is sometimes needed to avoid issues with DDP/FSDP if no visual inputs
            # Ensure this logic is robust and matches what Qwen expects for its visual model.
            # The original monkey_patch_forward.py has specific dummy handling.
            # For simplicity, we assume here that if pixel_values are None, they are truly not part of the batch.
            pass # Or insert refined dummy visual pass from monkey_patch_forward if strictly needed
            
        if pixel_values is not None and inputs_embeds.shape[0] > 0:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            if input_ids is not None and self.config.image_token_id is not None:
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None and inputs_embeds.shape[0] > 0:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw) # Assuming visual model handles videos
            if input_ids is not None and self.config.video_token_id is not None:
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None and inputs_embeds.shape[0] > 0:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # Step 2: Get RoPE index if needed (Simplified, actual implementation in Qwen is more complex)
        # This part is crucial and needs to be correct. For now, pass position_ids if available.
        # The original forward pass handles this with self.get_rope_index
        # We might need to call that if position_ids are not directly provided and cache_position is used.
        # For SFT, usually position_ids are not passed, and get_rope_index is called internally.
        
        # For simplicity, we assume the base model's forward handles RoPE index internally if position_ids is None
        # or we rely on the monkey-patched forward if this class is used in conjunction with it.
        # Let's call the original Qwen2_5_VLForConditionalGeneration's method to get transformer_outputs
        # This way, we leverage its full input processing including RoPE and monkey patching for Liger. 
        
        # This call will use the original lm_head if not careful.
        # We need the outputs from self.model (the Qwen2VLModel part)
        transformer_outputs = self.model(
            input_ids=None, # input_ids are already processed into inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position, # Qwen2.5 specific
            # second_per_grid_ts=second_per_grid_ts # Qwen2.5 specific, if applicable and passed in
        )
        hidden_states = transformer_outputs[0]

        # Use the hidden state of the last token for classification/regression
        # This is a common approach. Other pooling strategies (e.g., mean pooling)
        # or using a specific token's hidden state could also be considered.
        if hidden_states.shape[1] == 0: # handle empty sequence after processing
             # This can happen if inputs are completely masked, or after some processing. 
             # Create dummy outputs of the correct types and device to avoid errors downstream.
             # This is a safeguard; ideally, empty sequences should be filtered out earlier.
            batch_size = input_ids.shape[0] if input_ids is not None else (inputs_embeds.shape[0] if inputs_embeds is not None else 1)
            pooled_output = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=hidden_states.dtype)
        else:
            pooled_output = hidden_states[:, -1, :] 

        # Multi-head predictions
        type_logits = self.type_head(pooled_output)
        type_probs = torch.softmax(type_logits, dim=-1)

        outputs_cls = {}
        for task_name, head in self.classification_heads.items():
            outputs_cls[task_name] = head(pooled_output)

        output_reg = None
        if self.regression_head:
            output_reg = self.regression_head(pooled_output)

        # Loss calculation
        loss = None
        if task_id_labels is not None:
            loss_type_fct = CrossEntropyLoss()
            loss_type = loss_type_fct(type_logits.view(-1, len(self.task_types)), task_id_labels.view(-1))

            total_weighted_task_loss = torch.tensor(0.0, device=pooled_output.device, dtype=type_logits.dtype)
            
            cls_task_idx_counter = 0 # to index into classification_labels
            for i in range(len(self.task_types)):
                task_name = self.id_to_task_type[i]
                task_details = self.task_config[task_name]
                task_prob_for_loss = type_probs[:, i] # Use probabilities for weighting

                # Create a mask for samples belonging to the current task_id
                current_task_mask = (task_id_labels == i)
                if not current_task_mask.any():
                    continue # No samples for this task in the current batch

                if task_details["type"] == "classification":
                    cls_logits = outputs_cls[task_name][current_task_mask]
                    # classification_labels is [batch_size, num_total_classification_tasks_in_config]
                    # We need to pick the correct column for the current task_name
                    # Find the index of task_name among all classification tasks in TASK_CONFIG
                    current_cls_label_column = -1
                    temp_cls_idx = 0
                    for tn_iter, td_iter in self.task_config.items():
                        if td_iter['type'] == 'classification':
                            if tn_iter == task_name:
                                current_cls_label_column = temp_cls_idx
                                break
                            temp_cls_idx += 1
                    
                    if cls_logits.numel() > 0 and classification_labels is not None and current_cls_label_column != -1:
                        # Select labels for the current task and for the samples belonging to this task
                        # Assuming classification_labels has shape [batch_size, total_num_cls_tasks_defined_in_order]
                        # And only one of them is not IGNORE_INDEX per row.
                        # For simplicity, classification_labels should be [batch_size, num_categories_for_this_specific_task_type_if_active]
                        # This part of data structure for labels needs to be precise.
                        # Let's assume classification_labels is a single vector of labels for the active task.
                        # This requires data loader to prepare it this way.
                        # For now, using the pre-structured classification_labels: [batch_size, num_classification_tasks_overall]
                        # where only one column is active per sample.

                        # Let's adjust classification_labels to be: [batch_size], and it contains the label for the *active* classification task for that sample, or IGNORE_INDEX
                        # This matches how 'labels' is typically handled for CrossEntropyLoss with multiple classes. 
                        # The data loader needs to ensure `classification_labels` has the direct label for the task indicated by `task_id_labels` if it's a classification task.
                        
                        # Simplified: Assume classification_labels is [batch_size] and holds the target for the *active* classification task for this sample
                        # This means data.py must ensure `classification_labels[j]` is the label for task `task_id_labels[j]` if it's classification.
                        # This is a common pattern. If task_id_labels[j] is regression, then classification_labels[j] would be IGNORE_INDEX.
                        
                        if classification_labels is not None:
                            cls_targets = classification_labels[current_task_mask]
                            # Filter out IGNORE_INDEX from targets if loss function doesn't handle it for masked logits
                            valid_targets_mask = cls_targets != IGNORE_INDEX
                            if valid_targets_mask.any():
                                loss_cls_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                                loss_cls = loss_cls_fct(cls_logits[valid_targets_mask], cls_targets[valid_targets_mask])
                                if not torch.isnan(loss_cls) and not torch.isinf(loss_cls):
                                     total_weighted_task_loss += task_prob_for_loss[current_task_mask][valid_targets_mask].mean() * loss_cls # Weight by mean prob for valid samples
                    cls_task_idx_counter +=1

                elif task_details["type"] == "regression" and self.regression_head:
                    reg_preds = output_reg[current_task_mask].squeeze(-1)
                    if reg_preds.numel() > 0 and regression_labels is not None:
                        reg_targets = regression_labels[current_task_mask]
                        valid_targets_mask = reg_targets != IGNORE_INDEX # if using ignore_index for regression too
                        if valid_targets_mask.any():
                            loss_reg_fct = MSELoss()
                            loss_reg = loss_reg_fct(reg_preds[valid_targets_mask], reg_targets[valid_targets_mask])
                            if not torch.isnan(loss_reg) and not torch.isinf(loss_reg):
                                total_weighted_task_loss += task_prob_for_loss[current_task_mask][valid_targets_mask].mean() * loss_reg
            
            loss = loss_type + total_weighted_task_loss

        if not return_dict:
            # For compatibility with Trainer, logits is often expected. We give type_logits.
            # Other outputs can be part of a tuple.
            _outputs = (type_logits, outputs_cls, output_reg) + transformer_outputs[1:] # hidden_states, attentions etc from base model
            return (loss,) + _outputs if loss is not None else _outputs

        return MultiHeadQwenVLOutputWithPast(
            loss=loss,
            logits=type_logits, # Main logits for Trainer compatibility, can be type_logits
            past_key_values=transformer_outputs.past_key_values if hasattr(transformer_outputs, 'past_key_values') else None,
            hidden_states=transformer_outputs.hidden_states if hasattr(transformer_outputs, 'hidden_states') else None,
            attentions=transformer_outputs.attentions if hasattr(transformer_outputs, 'attentions') else None,
            # rope_deltas field might not exist in base Qwen2_5_VLModel's output, but in Qwen2_5_VLForConditionalGeneration's output
            # It's usually handled and returned by the final model class if use_cache=True and during generation
            # For training, it might not be directly part of transformer_outputs from self.model.
            # This needs to be checked against the actual return type of self.model(...)
            type_logits=type_logits,
            type_probs=type_probs,
            classification_outputs=outputs_cls,
            regression_output=output_reg,
        )

    # If this model is loaded using .from_pretrained, Hugging Face will try to load weights for all layers defined in __init__.
    # We need to ensure that if we are fine-tuning, these new heads are initialized.
    # If loading a previously fine-tuned multi-head model, their weights should be loaded.
    # The base Qwen model weights will be loaded by super().from_pretrained mechanism. 