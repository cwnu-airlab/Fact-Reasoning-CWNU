import torch
import transformers

class T5Classifier(transformers.T5ForConditionalGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.encoder = self.get_encoder()
		self.clf_head = torch.nn.Linear(config.d_model, config.num_class, bias=False)

	def forward(self, input_ids,
			**kwargs):
		encoder_output = self.encoder(input_ids)
		clf_logit = self.clf_head(encoder_output.last_hidden_state[:,-1,:])
		return transformers.modeling_outputs.SequenceClassifierOutput(
				logits=clf_logit,
				hidden_states=encoder_output,
				)

	def generate(self, input_ids,
			**kwargs):
		clf_logit = self.forward(input_ids).logits
		clf_index = torch.max(clf_logit, dim=1)[1]
		return clf_index
