import torch
import torch.nn.functional as F

from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext, apply_logit_softcap
from xtuner.v1.module import LMHead


def test_apply_logit_softcap():
    logits = torch.tensor([[-10.0, 0.0, 10.0]])
    softcap = 3.0

    expected = torch.tanh(logits / softcap) * softcap

    torch.testing.assert_close(apply_logit_softcap(logits, softcap), expected, rtol=0.0, atol=0.0)
    assert apply_logit_softcap(logits, None) is logits


def test_lm_head_applies_logit_softcap_without_loss_context():
    head = LMHead(4, 3, bias=False, logit_softcap=3.0)
    hidden_states = torch.randn(1, 2, 4)

    _, (actual, _) = head(hidden_states)
    expected = apply_logit_softcap(F.linear(hidden_states, head.weight), 3.0).float()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_ce_loss_applies_logit_softcap():
    hidden_states = torch.randn(2, 5, 4)
    head_weight = torch.randn(7, 4)
    labels = torch.randint(0, 7, (2, 5))
    labels[:, -1] = -100
    loss_weight = (labels != -100).float()
    loss_weight /= loss_weight.sum()

    logits = apply_logit_softcap(F.linear(hidden_states, head_weight), 3.0).float()
    expected = F.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
        reduction="none",
    )
    expected = (expected * loss_weight.flatten()).sum()

    for mode in ("eager", "chunk"):
        loss_ctx = LMHeadLossContext(
            CELossConfig(mode=mode, chunk_size=2, logit_softcap=3.0),
            CELossKwargs(shifted_labels=labels, loss_weight=loss_weight),
        )
        actual, _ = loss_ctx.forward(hidden_states, head_weight)
        torch.testing.assert_close(actual, expected)
