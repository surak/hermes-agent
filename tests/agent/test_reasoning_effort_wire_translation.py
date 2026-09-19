"""Wire translation for Hermes' extended reasoning-effort vocabulary (#89503).

Hermes' internal effort set extends the wire vocabulary with ``ultra`` (the
/reasoning command documents none..xhigh|max|ultra). OpenAI-compatible wires —
OpenRouter chief among them — accept exactly max|xhigh|high|medium|low|minimal|
none and reject the extension with HTTP 400:

    reasoning.effort: Invalid option: expected one of "max"|"xhigh"|"high"|
    "medium"|"low"|"minimal"|"none"

An ``ultra`` configured while the default model was Anthropic worked (the
Anthropic adapter maps its own levels), but the moment a per-job override
pinned an OpenRouter model the extension leaked through the OpenAI-compatible
transport untranslated and every call failed. ``_reasoning_config_for_model``
is the wire-compat chokepoint for this transport: it must cap the extension
for every model, not just the one vendor prefix that happened to be fixed
first.
"""

import pytest

from agent.transports.chat_completions import _reasoning_config_for_model


class TestUltraEffortWireTranslation:
    def test_ultra_maps_to_max_for_any_model(self):
        """The extension level caps at the wire vocabulary for every model —
        including the OpenRouter vendor prefixes a per-job override pins
        (#89503's nvidia/ case) and models with no vendor prefix at all."""
        for model in (
            "nvidia/nemotron-3.5-lightning:free",
            "deepseek/deepseek-v4",
            "qwen/qwen3.5-coder",
            "some-internal-model",
        ):
            out = _reasoning_config_for_model(
                model, {"enabled": True, "effort": "ultra"}
            )
            assert out == {"enabled": True, "effort": "max"}, model

    def test_gpt_56_ultra_still_maps(self):
        """The original pre-existing mapping (gpt-5.6 + ultra → max) is
        preserved by the generalized one."""
        out = _reasoning_config_for_model(
            "gpt-5.6", {"enabled": True, "effort": "ultra"}
        )
        assert out == {"enabled": True, "effort": "max"}

    def test_wire_native_levels_pass_through_untouched(self):
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cfg = {"enabled": True, "effort": level}
            assert _reasoning_config_for_model("any/model", cfg) == cfg

    def test_non_dict_and_none_pass_through(self):
        assert _reasoning_config_for_model("m", None) is None
        assert _reasoning_config_for_model("m", "not-a-dict") == "not-a-dict"


class TestKimiK3RelayVocabulary:
    """Custom OpenAI-compat relays fronting Kimi K3 (PTJ's gateway maps
    ``reasoning_effort`` onto Kimi's ``thinking_effort`` knob, which accepts
    only low/high/max) 400 on ``xhigh`` regardless of the client's hostname.
    The entry clamp must be keyed on the model slug, not the provider."""

    @pytest.mark.parametrize("model", ["k3", "Kimi-K3-1M", "kimi-k3-1m", "alias-kimi-k3-1m"])
    def test_xhigh_clamps_to_max(self, model):
        """Live-verified 2026-09-19: PTJ 400s with
        'Kimi K3 supports thinking_effort values: low, high, max
        (parameter=thinking_effort, value=xhigh)'."""
        out = _reasoning_config_for_model(model, {"enabled": True, "effort": "xhigh"})
        assert out == {"enabled": True, "effort": "max"}, model

    @pytest.mark.parametrize("model", ["kimi-k3", "k3-256k"])
    def test_medium_rounds_to_high(self, model):
        """K3's vendor mapping: medium is not a K3 tier — high is its
        positional middle (and server default)."""
        out = _reasoning_config_for_model(model, {"enabled": True, "effort": "medium"})
        assert out == {"enabled": True, "effort": "high"}, model

    def test_k3_native_tiers_pass_through(self):
        for level in ("low", "high", "max"):
            cfg = {"enabled": True, "effort": level}
            assert _reasoning_config_for_model("alias-kimi-k3-1m", cfg) == cfg

    def test_k3_ultra_maps_to_max(self):
        out = _reasoning_config_for_model("kimi-k3", {"enabled": True, "effort": "ultra"})
        assert out == {"enabled": True, "effort": "max"}

    def test_k2_era_slugs_keep_generic_wire(self):
        """K2-era Kimi speaks low/medium/high — the generic wire accepts
        xhigh verbatim (the relay fronts K2), so no clamp may change it."""
        cfg = {"enabled": True, "effort": "xhigh"}
        for model in ("kimi-k2.6", "moonshotai/kimi-k2.5"):
            assert _reasoning_config_for_model(model, cfg) == cfg, model
