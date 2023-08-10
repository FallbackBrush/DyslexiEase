import torchaudio
import torch
import matplotlib.pyplot as plt
import io

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to('cuda')
vocoder = bundle.get_vocoder().to('cuda')

audioprompt = "A man riding on a bicycle."

with torch.inference_mode():
    processed, lengths = processor(audioprompt)
    processed = processed.to('cuda')
    lengths = lengths.to('cuda')
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)
print("done")
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9))
ax1.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
ax2.plot(waveforms[0].cpu().detach())
buffer = io.BytesIO()
path = audioprompt+'.wav'
torchaudio.save(path,src=waveforms.detach().cpu(),sample_rate=22000)