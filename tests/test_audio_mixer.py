import importlib.util
import pathlib
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "ComfyUI/PyTorch runtime is not installed")
class AudioMixerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, DJ_AudioMixer
        import torch
        spec = importlib.util.spec_from_file_location(
            "autoeditor_audio_mixer_local", ROOT / "audio_mixer.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DJ_AudioMixer = module.DJ_AudioMixer

    def test_mix_preserves_longest_duration_and_reports_contract(self):
        first = {"waveform": torch.full((1, 1, 1000), 0.1), "sample_rate": 1000}
        second = {"waveform": torch.full((1, 1, 500), 0.2), "sample_rate": 1000}
        output, report = DJ_AudioMixer().mix_audio(
            first, second, 50, "start", limiter="disable", dc_offset_removal="disable"
        )
        self.assertEqual(tuple(output["waveform"].shape), (1, 1, 1000))
        self.assertEqual(output["sample_rate"], 1000)
        self.assertIn("Duration: 1.00s", report)
        self.assertIn("Sample rate: 1000Hz", report)

    def test_inputs_are_not_modified_in_place(self):
        waveform = torch.linspace(-0.2, 0.2, 100).reshape(1, 1, 100)
        original = waveform.clone()
        payload = {"waveform": waveform, "sample_rate": 1000}
        DJ_AudioMixer().mix_audio(payload, payload, 50, "start")
        self.assertTrue(torch.equal(payload["waveform"], original))


if __name__ == "__main__":
    unittest.main()
