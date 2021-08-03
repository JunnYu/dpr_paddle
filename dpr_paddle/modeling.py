from paddlenlp.transformers import PretrainedModel, register_base_model, BertModel
import paddle
import paddle.nn as nn
from typing import Optional

__all__ = [
    "DPREncoder",
    "DPRContextEncoder",
    "DPRQuestionEncoder",
    "DPRSpanPredictor",
    "DPRReader",
]


class DPRPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "dpr-ctx_encoder-multiset-base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "projection_dim": 0,
        },
        "dpr-ctx_encoder-single-nq-base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "projection_dim": 0,
        },
        "dpr-question_encoder-multiset-base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "projection_dim": 0,
        },
        "dpr-question_encoder-single-nq-base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "projection_dim": 0,
        },
        "dpr-reader-multiset-base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "projection_dim": 0,
        },
        "dpr-reader-single-nq-base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "projection_dim": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "dpr-ctx_encoder-single-nq-base": "",
            "dpr-ctx_encoder-single-nq-base": "",
            "dpr-question_encoder-multiset-base": "",
            "dpr-question_encoder-single-nq-base": "",
            "dpr-reader-multiset-base": "",
            "dpr-reader-single-nq-base": "",
        }
    }
    base_model_prefix = "bert_model"


@register_base_model
class DPREncoder(DPRPretrainedModel):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        pool_act="tanh",
        projection_dim=0,
    ):
        super().__init__()
        self.bert_model = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            pool_act=pool_act,
        )
        assert hidden_size > 0, "Encoder hidden_size can't be zero"

        self.projection_dim = projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(hidden_size, projection_dim)
        self.init_weights()

    def init_weights(self):
        self.bert_model.apply(self.bert_model.init_weights)
        if self.projection_dim > 0:
            self.encode_proj.apply(self.bert_model._init_weights)

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
    ):
        outputs = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        return sequence_output, pooled_output

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config["hidden_size"]


class DPRContextEncoder(DPRPretrainedModel):
    base_model_prefix = "ctx_encoder"

    def __init__(self, ctx_encoder):
        super().__init__()
        self.ctx_encoder = ctx_encoder
        self.init_weights()

    def init_weights(self):
        self.ctx_encoder.init_weights()

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
    ):
        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return outputs[1]


class DPRQuestionEncoder(DPRPretrainedModel):
    base_model_prefix = "question_encoder"

    def __init__(self, question_encoder):
        super().__init__()
        self.question_encoder = question_encoder
        self.init_weights()

    def init_weights(self):
        self.question_encoder.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs[1]


class DPRSpanPredictor(DPRPretrainedModel):
    base_model_prefix = "encoder"

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        pool_act="tanh",
        projection_dim=0,
    ):
        super().__init__()
        self.encoder = DPREncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            pool_act=pool_act,
            projection_dim=projection_dim,
        )
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        self.init_weights()

    def init_weights(self):
        self.encoder.init_weights()

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
    ):
        n_passages, sequence_length = input_ids.shape
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        relevance_logits = self.qa_classifier(sequence_output[:, 0])

        start_logits = start_logits.reshape(shape=[n_passages, sequence_length])
        end_logits = end_logits.reshape(shape=[n_passages, sequence_length])
        relevance_logits = relevance_logits.reshape(shape=[n_passages])

        return start_logits, end_logits, relevance_logits


class DPRReader(DPRPretrainedModel):

    base_model_class = DPRSpanPredictor
    base_model_prefix = "span_predictor"

    def __init__(self, span_predictor):
        super().__init__()
        self.span_predictor = span_predictor
        self.init_weights()

    def init_weights(self):
        self.span_predictor.encoder.init_weights()
        self.span_predictor.qa_classifier.apply(
            self.span_predictor.encoder.bert_model.init_weights
        )
        self.span_predictor.qa_outputs.apply(
            self.span_predictor.encoder.bert_model.init_weights
        )

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
    ):
        return self.span_predictor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
