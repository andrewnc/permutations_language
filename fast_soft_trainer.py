from transformers import Trainer
import tensorflow as tf
from perturbations.fenchel_young import FenchelYoungLoss

def ranks(inputs, axis=-1):
  """Returns the ranks of the input values among the given axis."""
  return 1 + tf.cast(
      tf.argsort(tf.argsort(inputs, axis=axis), axis=axis), dtype=inputs.dtype)


class FastSoftTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = FenchelYoungLoss(ranks)
        loss = loss_fn(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss