import torch
import transformers
import numpy as np

class T5Generator_with_pgn(transformers.T5ForConditionalGeneration):
	def __init__(self, config, **kwargs):
		super().__init__(config)
		self.encoder = self.get_encoder()
		self.decoder = self.get_decoder()

		self.gate = torch.nn.Linear(config.d_model, 1, bias=True)

	def get_attn_key_pad_mask(self, seq_k, seq_q, pad_idx):
		len_q = seq_q.size(1)
		padding_mask = seq_k.eq(pad_idx)
		padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1) # [batch, len_q, len_k]
		return padding_mask

	def forward(self,
			input_ids=None,
			labels=None,
			decoder_input_ids=None,
			decoder_inputs_embeds=None,
			encoder_outputs=None,
			past_key_values=None,
			attention_mask=None,
			decoder_attention_mask=None,
			head_mask=None,
			decoder_head_mask=None,
			inputs_embeds=None,
			use_cache=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None,
			pad_id=0,
			loss_func=torch.nn.CrossEntropyLoss(ignore_index=-100),
			**kwargs
			):

		if encoder_outputs is None:
			encoder_outputs = self.encoder(
				input_ids=input_ids,
			)
		elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
			encoder_outputs = BaseModelOutput(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
				attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
			)

		if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = self._shift_right(labels)

		hidden_states = encoder_outputs[0]

		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			inputs_embeds=decoder_inputs_embeds,
			past_key_values=past_key_values,
			encoder_hidden_states=hidden_states,
			encoder_attention_mask=attention_mask,
			head_mask=decoder_head_mask,
			#encoder_head_mask=head_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		
		sequence_output = decoder_outputs[0]
		sequence_output = sequence_output * (self.model_dim ** -0.5)

		lm_logits = self.lm_head(sequence_output)
		#super().forward(input_ids=input_ids, labels=labels)


		## PGN
		dec_enc_attn_mask = self.get_attn_key_pad_mask(seq_k=input_ids, seq_q=decoder_input_ids, pad_idx=pad_id)

		in_vocab_prob = torch.softmax(lm_logits, -1) # [batch, dec_len, vocab_size]
		copy_gate = torch.sigmoid(self.gate(sequence_output))
		
		full_vocab_prob = (1-copy_gate) * in_vocab_prob # [batch, dec_len, vocab_size]

		scores = torch.bmm(sequence_output, hidden_states.transpose(2,1)) # [batch, dec_len, enc_len]
		scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)
		
		oov_vocab_prob = torch.softmax(scores, -1)
		full_vocabl_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)
		# input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1) = [batch, dec_len, enc_len]

		pgn_logits = torch.log(full_vocab_prob + 1e-8)

		loss = None
		if labels is not None:
			loss = loss_func(pgn_logits.view(-1, pgn_logits.size(-1)), labels.view(-1))

		return transformers.modeling_outputs.Seq2SeqLMOutput(
			loss=loss,
			logits=pgn_logits,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)
		
	@torch.no_grad()
	def generate(
		self,
		input_ids=None,
		attention_mask=None,
		max_length=None,
		min_length=None,
		do_sample=None,
		early_stopping=None,
		num_beams=None,
		temperature=None,
		top_k=None,
		top_p=None,
		repetition_penalty=None,
		bad_words_ids=None,
		bos_token_id=None,
		pad_token_id=0,
		eos_token_id=1,
		length_penalty=None,
		no_repeat_ngram_size=None,
		encoder_no_repeat_ngram_size=None,
		num_return_sequences=None,
		max_time=None,
		decoder_start_token_id=None,
		use_cache=None,
		num_beam_groups=None,
		diversity_penalty=None,
		prefix_allowed_tokens_fn=None,
		output_attentions=None,
		output_hidden_states=None,
		output_scores=None,
		return_dict_in_generate=None,
		forced_bos_token_id=None,
		forced_eos_token_id=None,
		remove_invalid_values=None,
		**model_kwargs,
		):

		if max_length==None: max_length = self.config.max_length
		
		decoder_input_ids = torch.tensor([[pad_token_id]]*input_ids.shape[0]).to(self.device)

		encoder_outputs = None
		for i in range(max_length):
			model_output = self.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, attention_mask=attention_mask)
			if not encoder_outputs:
				encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
						last_hidden_state = model_output.encoder_last_hidden_state,
						hidden_states = model_output.encoder_hidden_states,
						attentions = model_output.encoder_attentions
						)

			logits = model_output.logits.detach()
			predict = torch.argmax(logits, dim=-1)[:,-1:]

			decoder_input_ids = torch.cat((decoder_input_ids, predict),1)

		predict = decoder_input_ids
		return predict 
